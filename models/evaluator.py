"""Run simple cell optimisation"""

import os
import json
import pickle
import pathlib
import bluepyopt.ephys as ephys

import logging

import model

logger = logging.getLogger("__main__")

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def get_protocol_definitions(model):
    """
    Returns protocol definitions

    Parameters
    ----------
    model: str
        "hay", "hallerman"

    Returns
    -------
    protocols_dict: dict
        Dictionary with protocol definitions
    """
    path_protocols = pathlib.Path(f"{model}_model") / "protocols.json"

    return json.load(open(path_protocols))


def get_feature_definitions(feature_file, feature_set=None):
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
    
    if ".pkl" in feature_file:
        feature_definitions = pickle.load(open(feature_file, 'rb'))
    elif ".json" in feature_file:
        feature_definitions = json.load(open(feature_file))
    else:
        raise Exception("feature_file is neither or pickle nor a json file.")

    if feature_set:
        return feature_definitions[feature_set]

    return feature_definitions


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

    if electrode:
        recordings.append(
            ephys.recordings.LFPRecording("%s.MEA.LFP" % protocol_name)
        )

    return recordings


def define_stimuli_hay(protocol_name, protocol_definition):
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


def define_stimuli_hallerman(protocol_name, protocol_definition):
    return []
    
    
def define_protocols(
    model,
    feature_set=None,
    feature_file=None,
    electrode=None,
    protocols_with_lfp=None
):
    """
    Defines protocols for a specified feature_Set (or file)

    Parameters
    ----------
    model: str
        "hay", "hallerman"
    feature_set: str
        "soma", "multiple", "extra", or "all"
    feature_file: str
        Path to a feature pkl file to load protocols from
    electrode: LFPy.RecExtElectrode
        If given, the MEA recording is added
    protocols_with_lfp: list or None
        List of protocols for which LFP should be computed. If None, LFP are
        added to all protocols.

    Returns
    -------
    protocols: dict
        Dictionary with defined protocols
    """

    protocol_definitions = get_protocol_definitions(model)
    feature_definitions = get_feature_definitions(feature_file, feature_set)

    protocols = {}

    if protocols_with_lfp is None:
        protocols_with_lfp = []

    for protocol_name in feature_definitions:

        if protocol_name in protocols_with_lfp:
            recordings = define_recordings(
                protocol_name, protocol_definitions[protocol_name], electrode
            )
        else:
            recordings = define_recordings(
                protocol_name, protocol_definitions[protocol_name], None
            )
        
        if model == 'hay':
            stimuli = define_stimuli_hay(
                protocol_name, protocol_definitions[protocol_name]
            )
        else:
            stimuli = define_stimuli_hallerman(
                protocol_name, protocol_definitions[protocol_name]
            )
            
        protocols[protocol_name] = ephys.protocols.SweepProtocol(
            protocol_name, stimuli, recordings, cvode_active=True
        )

    return protocols


def get_release_params(model):
    """
    Returns release params for the hay model

    Parameters
    ----------
    model: str
        "hay", "hallerman", "extra"

    Returns
    -------
    release_params: dict
        Dictionary with parameters and their release values
    """

    # load release params
    release_params_file = pathlib.Path(f"{model}_model") / \
        "parameters_release.json"

    # load unfrozen params
    params_file = pathlib.Path(f"{model}_model") / "parameters.json"

    all_release_params = {}
    with open(release_params_file, 'r') as f:

        data = json.load(f)

        for prm in data:
            all_release_params[f"{prm['param_name']}.{prm['sectionlist']}"] = \
                prm["value"]

    params_bounds = {}
    with open(params_file, 'r') as f:

        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = \
                    prm["bounds"]

    release_params = {}
    for k, v in all_release_params.items():
        if k in params_bounds.keys():
            release_params[k] = v

    return release_params


def get_unfrozen_params_bounds(model):
    """
    Returns unfrozen params bounds model

    Parameters
    ----------
    model: str
        "hay", "hallerman", "extra"

    Returns
    -------
    params_bounds: dict
        Dictionary with parameters and their bounds
    """
    params_file = pathlib.Path(f"{model}_model") / "parameters.json"

    params_bounds = {}
    with open(params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = \
                    prm["bounds"]

    return params_bounds


def define_fitness_calculator(
    protocols, feature_file, feature_set, probe=None
):
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
    probe: MEAutility.MEA
        The probe to use for extracellular features

    Returns
    -------

    """

    if feature_set not in ['multiple', 'soma', 'extra']:
        raise Exception("feature_set should be 'multiple', 'soma' or 'extra'.")

    if feature_set == 'extra' and probe is None:
        raise Exception("Provide a MEAutility probe to use the 'extra' set.")

    feature_definitions = get_feature_definitions(feature_set, feature_file)

    objectives = []
    efeatures = {}

    for protocol_name, locations in feature_definitions.items():

        efeatures[protocol_name] = []

        for location, features in locations.items():

            for efel_feature_name, meanstd in features.items():

                feature_name = '%s.%s.%s' % (protocol_name, location, efel_feature_name)

                stimulus = protocols[protocol_name].stimuli[0]

                kwargs = {
                    'exp_mean': meanstd[0],
                    'exp_std': meanstd[1],
                    'stim_start': stimulus.step_delay
                }

                if location == 'soma':
                    kwargs['threshold'] = -20
                elif 'dend' in location:
                    kwargs['threshold'] = -55
                else:
                    kwargs['threshold'] = -20

                if protocol_name == 'bAP':
                    kwargs['stim_end'] = stimulus.total_duration
                else:
                    kwargs[ 'stim_end'] = stimulus.step_delay + stimulus.step_duration

                if location == 'MEA':

                    feature = ephys.efeatures.extraFELFeature(
                        name=feature_name,
                        extrafel_feature_name=efel_feature_name,
                        recording_names={'': '%s.%s.LFP' % (protocol_name, location)},
                        somatic_recording_name=f'{protocol_name}.soma.v',
                        channel_locations=probe.positions,
                        channel_id=None,
                        fs=20,
                        fcut=1,
                        ms_cut=[3, 10],
                        upsample=10,
                        **kwargs
                    )

                else:

                    feature = ephys.efeatures.eFELFeature(
                        name=feature_name,
                        efel_feature_name=efel_feature_name,
                        recording_names={'': '%s.%s.v' % (protocol_name,
                                                          location)},
                        **kwargs
                    )

                efeatures[protocol_name].append(feature)

                objectives.append(
                    ephys.objectives.SingletonObjective(
                        feature_name,
                        feature
                    )
                )

    return ephys.objectivescalculators.ObjectivesCalculator(
        objectives), efeatures


def create_evaluator(
        model,
        feature_set,
        sample_id,
        morph_modifier=""
):
    """
        Prepares objects for optimization of test models.
        Features are assumed to be pkl files in 'config_dir'/random_'sample_id'
            /'feature_set'.pkl

        Parameters
        ----------
        model: str
            "hay" or "hallerman"
        feature_set: str
            "soma", "multiple", "extra", or "all"
        sample_id: int
            The test sample ID
        morph_modifier: str
            The modifier to apply to the axon:
                - "hillock": the axon is replaced with an axon hillock, an AIS,
                and a myelinated linear axon.
                   The hillock morphology uses the original axon reconstruction.
                   The 'axon', 'ais', 'hillock', and 'myelin' sections are added
                - "taper": the axon is replaced with a tapered hillock
                - "": the axon is replaced by a 2-segment axon stub

        Returns
        -------
        CellEvaluator
        """

    sample_dir = pathlib.Path(f"{model}_model") / 'features' / f'random_{sample_id}'

    probe = None
    electrode = None
    if feature_set == "extra":
        probe_file = sample_dir / 'probe.json'
        probe, electrode = model.define_electrode(probe_file=probe_file)

    feature_file = sample_dir / f'{feature_set}.pkl'
    fitness_protocols = define_protocols(feature_set, feature_file, electrode=electrode)

    cell = model.create(model, morph_modifier, release=False)

    sim = ephys.simulators.LFPySimulator(
        LFPyCellModel=cell, cvode_active=True, electrode=electrode
    )

    fitness_calculator, _ = define_fitness_calculator(
        protocols=fitness_protocols,
        feature_file=feature_file,
        feature_set=feature_set,
        probe=probe,
    )

    param_names = [
        param.name for param in cell.params.values() if not param.frozen
    ]

    return ephys.evaluators.CellEvaluator(
        cell_model=cell,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=sim,
        timeout=900
    )
