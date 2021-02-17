"""Run simple cell optimisation"""

import os
import json
import pickle
from pathlib import Path

import bluepyopt as bpopt
import bluepyopt.ephys as ephys

import logging
import model

from configs import config_dir

logger = logging.getLogger("__main__")

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def get_protocol_definitions():
    """
    Returns protocol definitions

    Returns
    -------
    protocols_dict: dict
        Dictionary with protocol definitions
    """
    return json.load(open(os.path.join(config_dir, "protocols.json")))


def get_feature_definitions(feature_set=None, feature_file=None):
    """
    Returns features definitions

    Parameters
    ----------
    feature_set: str
        "soma", "multiple", "extra", or "all"
    feature_file: str
        Path to json file specifying list of features for each fetaure set

    Returns
    -------
    fetaures_dict: dict
        Dictionary with features definitions
    """

    if feature_file is not None:
        assert feature_set is not None
        return pickle.load(open(feature_file, 'rb'))
    else:
        if feature_set is not None:
            return json.load(open(os.path.join(config_dir, 'features_list.json')))[feature_set]
        else:
            return json.load(open(os.path.join(config_dir, 'features_list.json')))["multiple"]


def define_recordings(protocol_name, protocol_definition, electrode=None):
    """
    Defines recordings for the specified protocol

    Parameters
    ----------
    protocol_name: str
        The protocol name
    protocol_definition: dict
        Dictionary with protocol definitions (used to access "extra_recordings")
    electrode: LFPy.RecExtElectrode
        If given, the MEA recording is added

    Returns
    -------
    recording: list
        List of defined recordings for the specified protocol_name
    """
    recordings = [
        ephys.recordings.CompRecording(
            name="%s.soma.v" % protocol_name, location=soma_loc, variable="v"
        )
    ]

    if "extra_recordings" in protocol_definition:
        for recording_definition in protocol_definition["extra_recordings"]:

            if recording_definition["type"] == "somadistance":
                location = ephys.locations.NrnSomaDistanceCompLocation(
                    name=recording_definition["name"],
                    soma_distance=recording_definition["somadistance"],
                    seclist_name=recording_definition["seclist_name"],
                )

                var = recording_definition["var"]
                recordings.append(
                    ephys.recordings.CompRecording(
                        name="%s.%s.%s" % (protocol_name, location.name, var),
                        location=location,
                        variable=recording_definition["var"],
                    )
                )
            elif recording_definition['type'] == 'nrnseclistcomp':
                location = ephys.locations.NrnSeclistCompLocation(
                    name=recording_definition['name'],
                    comp_x=recording_definition['comp_x'],
                    sec_index=recording_definition['sec_index'],
                    seclist_name=recording_definition['seclist_name'])

                var = recording_definition["var"]
                recordings.append(
                    ephys.recordings.CompRecording(
                        name="%s.%s.%s" % (protocol_name, location.name, var),
                        location=location,
                        variable=recording_definition["var"],
                    )
                )
            else:
                raise Exception("Type not supported")
    
    ############## HACK ##############
    #for d in [15]:
    #    location = ephys.locations.NrnSomaDistanceCompLocation(
    #        name=f"hillock_{d}",
    #        soma_distance=d,
    #        seclist_name="hillockal",
    #    )
    #    recordings.append(
    #        ephys.recordings.CompRecording(
    #            name="%s.%s.%s" % (protocol_name, location.name, "v"),
    #            location=location,
    #            variable="v",
    #        )
    #    )
    #
    #for d in [35, 45, 55]:
    #    location = ephys.locations.NrnSomaDistanceCompLocation(
    #        name=f"ais_{d}",
    #        soma_distance=d,
    #        seclist_name="axon_initial_segment",
    #    )
    #    recordings.append(
    #        ephys.recordings.CompRecording(
    #            name="%s.%s.%s" % (protocol_name, location.name, "v"),
    #            location=location,
    #            variable="v",
    #        )
    #    )

    if electrode is not None:
        recordings.append(ephys.recordings.LFPRecording("%s.MEA.LFP" % protocol_name))

    return recordings


def define_stimuli(protocol_name, protocol_definition):
    """
    Defines stimuli associated with a specified protocol

    Parameters
    ----------
    protocol_name: str
        The protocol name
    protocol_definition: dict
        Dictionary with protocol definitions (used to access "extra_recordings")

    Returns
    -------
    stimuli: list
        List of defined stimuli for the specified protocol_name

    """
    stimuli = []
    for stimulus_definition in protocol_definition["stimuli"]:

        if protocol_name in ['BAC', 'Step1', 'Step2', 'Step3', 'bAP']:
            stimuli.append(ephys.stimuli.LFPySquarePulse(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=soma_loc,
                total_duration=stimulus_definition['totduration']))

        if protocol_name in ['EPSP', 'BAC']:
            loc_api = ephys.locations.NrnSomaDistanceCompLocation(
                name=protocol_name,
                soma_distance=620,
                seclist_name="apical",
            )

            stimuli.append(ephys.stimuli.LFPySquarePulse(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=loc_api,
                total_duration=stimulus_definition['totduration']))

        if protocol_name in ['CaBurst']:
            loc_api = ephys.locations.NrnSomaDistanceCompLocation(
                name=protocol_name,
                soma_distance=620,
                seclist_name="apical",
            )

            stimuli.append(ephys.stimuli.LFPySquarePulse(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=loc_api,
                total_duration=stimulus_definition['totduration']))

    return stimuli


def define_protocols(feature_set=None, feature_file=None, electrode=None,
                     protocols_with_lfp=None):
    """
    Defines protocols for a specified feature_Set (or file)

    Parameters
    ----------
    feature_set: str
        "soma", "multiple", "extra", or "all"
    feature_file: str
        Path to a feature pkl file to load protocols from
    electrode: LFPy.RecExtElectrode
        If given, the MEA recording is added
    protocols_with_lfp: list or None
        List of protocols for which LFP should be computed. If None, LFP are added to all protocols

    Returns
    -------
    protocols: dict
        Dictionary with defined protocols
    """
    protocol_definitions = get_protocol_definitions()
    feature_definitions = get_feature_definitions(feature_set, feature_file)

    protocols = {}

    for protocol_name in feature_definitions:

        if protocols_with_lfp is not None:
            if protocol_name in protocols_with_lfp:
                recordings = define_recordings(protocol_name, protocol_definitions[protocol_name], electrode)
            else:
                recordings = define_recordings(protocol_name, protocol_definitions[protocol_name], None)
        else:
            recordings = define_recordings(protocol_name, protocol_definitions[protocol_name], electrode)
        stimuli = define_stimuli(protocol_name, protocol_definitions[protocol_name])

        protocols[protocol_name] = ephys.protocols.SweepProtocol(
            protocol_name, stimuli, recordings, cvode_active=True
        )

    return protocols


def get_release_params():
    """
    Returns release params for the hay model

    Returns
    -------
    release_params: dict
        Dictionary with parameters and their release values
    """
    # load release params
    release_params_file = os.path.join(config_dir, "parameters_release.json")
    # load unfrozen params
    params_file = os.path.join(config_dir, "parameters.json")

    all_release_params = {}
    with open(release_params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            all_release_params[f"{prm['param_name']}.{prm['sectionlist']}"] = prm["value"]

    params_bounds = {}
    with open(params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = prm["bounds"]

    release_params = {}
    for k, v in all_release_params.items():
        if k in params_bounds.keys():
            release_params[k] = v

    return release_params


def get_unfrozen_params_bounds():
    """
    Returns unfrozen params bounds model

    Returns
    -------
    params_bounds: dict
        Dictionary with parameters and their bounds
    """
    # load unfrozen params
    params_file = os.path.join(config_dir, "parameters.json")

    params_bounds = {}
    with open(params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = prm["bounds"]

    return params_bounds


def define_fitness_calculator(protocols, feature_file, feature_set, channels="map", probe=None):
    """
    Defines objective calculator

    Parameters
    ----------
    protocols: dict
        Dictionary with defined protocols
    feature_file: str
        Path to json file specifying list of features for each feature set
    feature_set: str
        "soma", "multiple", "extra", or "all"
    channels: list, "map", or None
        If None, features are computed separately for each channel
        If list, features are computed separately for the provided channels
        If 'map' (default), each feature is an array with the features computed on all channels
    probe: MEAutility.MEA
        The probe to use for extracellular features

    Returns
    -------

    """
    assert feature_set in ['multiple', 'soma', 'extra']
    if feature_set == 'extra':
        assert probe is not None, "Provide a MEAutility probe to use the 'extra' set"

    feature_definitions = get_feature_definitions(feature_set, feature_file)

    objectives = []
    efeatures = {}

    for protocol_name, locations in feature_definitions.items():

        efeatures[protocol_name] = []

        for location, features in locations.items():

            for efel_feature_name, meanstd in features.items():

                feature_name = '%s.%s.%s' % (protocol_name, location, efel_feature_name)

                stimulus = protocols[protocol_name].stimuli[0]

                kwargs = {'stim_start': stimulus.step_delay}

                if location == 'soma':
                    kwargs['threshold'] = -20
                elif 'dend' in location:
                    kwargs['threshold'] = -55
                else:
                    kwargs['threshold'] = -20

                if protocol_name == 'bAP':
                    kwargs['stim_end'] = stimulus.total_duration
                else:
                    kwargs['stim_end'] = stimulus.step_delay + stimulus.step_duration

                if location == 'MEA':
                    feature_class = ephys.efeatures.extraFELFeature
                    kwargs['recording_names'] = {'': '%s.%s.LFP' % (protocol_name, location)}
                    kwargs['fs'] = 20
                    kwargs['fcut'] = 1
                    kwargs['ms_cut'] = [3, 10]
                    kwargs['upsample'] = 10
                    kwargs['somatic_recording_name'] = f'{protocol_name}.soma.v'
                    kwargs['channel_locations'] = probe.positions

                    if channels == 'map':
                        kwargs['channel_id'] = None
                        kwargs['extrafel_feature_name'] = efel_feature_name

                    else:
                        channel_id = int(efel_feature_name.split('_')[-1])
                        kwargs['extrafel_feature_name'] = '_'.join(efel_feature_name.split('_')[:-1])
                        if channels is not None and channel_id not in channels:
                            continue
                        else:
                            kwargs['channel_id'] = channel_id

                else:
                    feature_class = ephys.efeatures.eFELFeature
                    kwargs['efel_feature_name'] = efel_feature_name
                    kwargs['recording_names'] = {'': '%s.%s.v' % (protocol_name, location)}

                feature = feature_class(
                    feature_name,
                    exp_mean=meanstd[0],
                    exp_std=meanstd[1],
                    **kwargs
                )

                efeatures[protocol_name].append(feature)

                objectives.append(
                    ephys.objectives.SingletonObjective(
                        feature_name,
                        feature
                    )
                )

    return ephys.objectivescalculators.ObjectivesCalculator(objectives), efeatures


def prepare_optimization(feature_set, sample_id, offspring_size=10, channels='map', map_function=None,
                         optimizer="CMA", seed=1, morph_modifier=""):
    """
    Prepares objects for optimization of test models with CMA or IBEA.
    Features are assumed to be pkl files in 'config_dir'/random_'sample_id'/'feature_set'.pkl

    Parameters
    ----------
    feature_set: str
        "soma", "multiple", "extra", or "all"
    sample_id: int
        The test sample ID
    offspring_size: int
        Offspring size
    channels: list, "map", or None
        If None, features are computed separately for each channel
        If list, features are computed separately for the provided channels
        If 'map' (default), each feature is an array with the features computed on all channels
    map_function: mapper
        Map function to be used by the optimizer
    seed: int
        The random seed
    morph_modifier: str
        The modifier to apply to the axon:
            - "hillock": the axon is replaced with an axon hillock, an AIS, and a myelinated linear axon.
               The hillock morphology uses the original axon reconstruction. The 'axon',
              'ais', 'hillock', and 'myelin' sections are added.
            - "taper": the axon is replaced with a tapered hillock
            - "": the axon is replaced by a 2-segment axon stub

    Returns
    -------
    opt_dict: dict
        A dictionary with the optimization objects
            - 'optimisation': opt
            - 'evaluator': evaluator
            - 'objectives_calculator': fitness_calculator
            - 'protocols': fitness_protocols
    """
    config_path = Path(config_dir)
    feature_file = config_path / 'features' / f'random_{sample_id}' / f'{feature_set}.pkl'

    probe = None
    electrode = None

    if feature_set == "extra" and channels == 'map':
        probe_file = config_path / 'features' / f'random_{sample_id}' / 'probe.json'
        probe, electrode = model.define_electrode(probe_file=probe_file)

    cell = model.create(morph_modifier, release=False)

    param_names = [param.name for param in cell.params.values() if not param.frozen]

    fitness_protocols = define_protocols(feature_set, feature_file, electrode=electrode)

    sim = ephys.simulators.LFPySimulator(LFPyCellModel=cell, cvode_active=True, electrode=electrode)

    fitness_calculator, _ = define_fitness_calculator(
        protocols=fitness_protocols,
        feature_file=feature_file,
        feature_set=feature_set,
        probe=probe,
        channels=channels
    )

    evaluator = ephys.evaluators.CellEvaluator(cell_model=cell,
                                               param_names=param_names,
                                               fitness_protocols=fitness_protocols,
                                               fitness_calculator=fitness_calculator,
                                               sim=sim,
                                               timeout=900)

    if optimizer == "CMA":
        opt = bpopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            offspring_size=offspring_size,
            seed=seed,
            map_function=map_function,
            weight_hv=0.4,
            selector_name="multi_objective"
        )
    elif optimizer == "IBEA":
        opt = bpopt.optimisations.DEAPOptimisation(evaluator=evaluator,
                                                   offspring_size=offspring_size,
                                                   seed=seed,
                                                   map_function=map_function)

    return {'optimisation': opt, 'evaluator': evaluator, 'objectives_calculator': fitness_calculator,
            'protocols': fitness_protocols}


def run_optimization(feature_set, sample_id, opt, channels, max_ngen):
    """
    Runs the optimization after preparing optimization objects.
    Checkpoints are saved in optimizaiton_results/checkpoints/random_'sample_id'/
                             'feature_set'_off'offspring_size'_ngen'max_ngen'_'nchannels'chan.pkl'

    Parameters
    ----------
    feature_set: str
        "soma", "multiple", "extra", or "all"
    sample_id: int
        The test sample ID
    opt: BPO optimizer
        The optimizer to be used
    channels: list, "map", or None
        If None, features are computed separately for each channel
        If list, features are computed separately for the provided channels
        If 'map' (default), each feature is an array with the features computed on all channels
    max_ngen: int
        Maximum number of generations

    Returns
    -------
    opt_results: dict
        Dictionary with optimization results:
            - 'final_pop': final_pop
            - 'halloffame': halloffame
            - 'log': log
            - 'hist': hist
    """
    if channels is None:
        nchannels = 'all'
    elif channels == 'map':
        nchannels = f'map'
    else:
        nchannels = len(channels)

    cp_filename = Path('optimization_results') / 'checkpoints' / f'random_{sample_id}' / \
                  f'{feature_set}_off{opt.offspring_size}_ngen{max_ngen}_{nchannels}chan.pkl'

    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False

    final_pop, halloffame, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_filename, continue_cp=continue_cp)

    return {'final_pop': final_pop, 'halloffame': halloffame, 'log': log, 'hist': hist}
