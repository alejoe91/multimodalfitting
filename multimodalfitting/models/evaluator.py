"""Run simple cell optimisation"""

import json
import pickle
import pathlib
from pathlib import Path
import numpy as np

import bluepyopt.ephys as ephys

import logging

from .model import define_electrode, create_ground_truth_model, create_experimental_model

logger = logging.getLogger("__main__")

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def get_protocol_definitions(model_name, protocols_file=None):
    """
    Returns protocol definitions

    Parameters
    ----------
    model_name: str
        "hay", "hallermann"
    protocols_file: str/Path
        If given, protocols are loaded from the json file

    Returns
    -------
    protocols_dict: dict
        Dictionary with protocol definitions
    """

    if protocols_file is None:
        path_protocols = pathlib.Path(f"{model_name}_model") / "protocols.json"
    else:
        assert Path(protocols_file).is_file(), "The protocols file does not exist"
        assert Path(protocols_file).suffix == ".json", "The protocols file must be a json file"
        path_protocols = protocols_file

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
    feature_file = Path(feature_file)

    if ".pkl" in feature_file.name:
        feature_definitions = pickle.load(open(feature_file, 'rb'))
    elif ".json" in feature_file.name:
        feature_definitions = json.load(open(feature_file))
    else:
        raise Exception("feature_file is neither or pickle nor a json file.")

    if feature_set:
        return feature_definitions[feature_set]

    return feature_definitions


def define_recordings(protocol_name, protocol_definition, electrode=None, extra_recordings=None):
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

            elif recording_definition["type"] == "nrnseclistcomp":
                location = ephys.locations.NrnSeclistCompLocation(
                    name=recording_definition["name"],
                    comp_x=recording_definition["comp_x"],
                    seclist_name=recording_definition["seclist_name"],
                    sec_index=recording_definition["sec_index"]
                )

            var = recording_definition["var"]
            recordings.append(
                ephys.recordings.CompRecording(
                    name="%s.%s.%s" % (protocol_name, location.name, var),
                    location=location,
                    variable=recording_definition["var"],
                )
            )

    if extra_recordings is not None:
        # check for substring
        if protocol_name in extra_recordings:
            for recording_definition in extra_recordings[protocol_name]:

                if recording_definition["type"] == "somadistance":
                    location = ephys.locations.NrnSomaDistanceCompLocation(
                        name=recording_definition["name"],
                        soma_distance=recording_definition["somadistance"],
                        seclist_name=recording_definition["seclist_name"],
                    )

                elif recording_definition["type"] == "nrnseclistcomp":
                    location = ephys.locations.NrnSeclistCompLocation(
                        name=recording_definition["name"],
                        comp_x=recording_definition["comp_x"],
                        seclist_name=recording_definition["seclist_name"],
                        sec_index=recording_definition["sec_index"]
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


def define_stimuli(protocol_definition, simulator="lfpy"):
    """
    Defines stimuli associated with a specified protocol

    Parameters
    ----------
    protocol_definition: dict
        Dictionary with protocol definitions (used to access "extra_recordings")

    Returns
    -------
    stimuli: list
        List of defined stimuli for the specified protocol_name

    """
    stimuli = []

    if simulator == "lfpy":
        stimulus_class = ephys.stimuli.LFPySquarePulse
    else:
        stimulus_class = ephys.stimuli.NrnSquarePulse

    for stimulus_definition in protocol_definition["stimuli"]:
        stimuli.append(stimulus_class(
            step_amplitude=stimulus_definition['amp'],
            step_delay=stimulus_definition['delay'],
            step_duration=stimulus_definition['duration'],
            location=soma_loc,
            total_duration=stimulus_definition['totduration']))

    return stimuli


def define_protocols(
        model_name,
        feature_set=None,
        feature_file=None,
        protocols_file=None,
        electrode=None,
        protocols_with_lfp=None,
        extra_recordings=None,
        simulator="lfpy"
):
    """
    Defines protocols for a specified feature_Set (or file)

    Parameters
    ----------
    model_name: str
        "hay", "hallermann", "cultured"
    feature_set: str
        "soma", "multiple", "extra", or "all"
    feature_file: str
        Path to a feature json file to load protocols from
    protocols_file: str
        Path to a feature json file to load protocols from
    electrode: LFPy.RecExtElectrode
        If given, the MEA recording is added
    protocols_with_lfp: list or None
        List of protocols for which LFP should be computed. If None, LFP are
        added to all protocols.
    simulator : str, optional
        The simulator to use. "lfpy" | "neuron"

    Returns
    -------
    protocols: dict
        Dictionary with defined protocols
    """
    protocol_definitions = get_protocol_definitions(model_name, protocols_file)
    feature_definitions = get_feature_definitions(feature_file, feature_set)

    assert simulator.lower() in ["lfpy", "neuron"]

    protocols = {}

    if protocols_with_lfp is None:
        protocols_with_lfp = []

    for protocol_name in feature_definitions:

        if protocol_name in protocols_with_lfp:
            recordings = define_recordings(
                protocol_name, protocol_definitions[protocol_name], electrode, extra_recordings
            )
        else:
            recordings = define_recordings(
                protocol_name, protocol_definitions[protocol_name], None, extra_recordings
            )

        stimuli = define_stimuli(protocol_definitions[protocol_name], simulator=simulator)

        protocols[protocol_name] = ephys.protocols.SweepProtocol(
            protocol_name, stimuli, recordings, cvode_active=True
        )

    return protocols


def define_test_step_protocol(step_amplitude=0.5, tot_duration=500, delay=50,
                              step_duration=400, probe=None, protocol_name="TestStep",
                              simulator="lfpy"):
    """Generates test protocol with a current pulse.

    Parameters
    ----------
    step_amplitude : float, optional
        Amplitude of the current pulse, by default 0.5
    tot_duration : int, optional
        Total duration of the protocol in ms, by default 500
    delay : int, optional
        Delay from the stimulus onset in ms, by default 50
    step_duration : int, optional
        Duration of the step in ms, by default 400
    probe : MEAutility.MEA, optional
        The extracellular probe, by default None
    protocol_name : str, optional
        The protocol name, by default "TestStep"
    simulator : str, optional
        The simulator to use. "lfpy" | "neuron"

    Returns
    -------
    protocols: dict
        Dictionary of BluePyOpt protocols
    """

    protocol_definition = {
        "stimuli": [
          {
            "delay": 0.0,
            "amp": 0.0,
            "duration": tot_duration,
            "totduration": tot_duration
          },
          {
            "delay": delay,
            "amp": step_amplitude,
            "duration": step_duration,
            "totduration": tot_duration
          }
        ]
    }
    recordings = define_recordings(protocol_name, protocol_definition, probe)

    stimuli = define_stimuli(protocol_definition)

    protocol = ephys.protocols.SweepProtocol(
        protocol_name, stimuli, recordings, cvode_active=True
    )

    protocols = {protocol_name: protocol}

    return protocols


def get_release_params(model_name):
    """
    Returns release params for the hay model

    Parameters
    ----------
    model_name: str
        "hay", "hallermann", "extra"

    Returns
    -------
    release_params: dict
        Dictionary with parameters and their release values
    """

    # load release params
    release_params_file = pathlib.Path(f"{model_name}_model") / "parameters_release.json"

    # load unfrozen params
    params_file = pathlib.Path(f"{model_name}_model") / "parameters.json"

    all_release_params = {}
    with open(release_params_file, 'r') as f:

        data = json.load(f)

        for prm in data:
            if prm["type"] != "global":
                all_release_params[f"{prm['param_name']}_{prm['sectionlist']}"] = \
                    prm["value"]

    params_bounds = {}
    with open(params_file, 'r') as f:

        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}_{prm['sectionlist']}"] = \
                    prm["bounds"]

    release_params = {}
    for k, v in all_release_params.items():
        if k in params_bounds.keys():
            release_params[k] = v

    return release_params


def get_unfrozen_params_bounds(model_name):
    """
    Returns unfrozen params bounds model

    Parameters
    ----------
    model_name: str
        "hay", "hallermann", "extra"

    Returns
    -------
    params_bounds: dict
        Dictionary with parameters and their bounds
    """
    params_file = pathlib.Path(f"{model_name}_model") / "parameters.json"

    params_bounds = {}
    with open(params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = \
                    prm["bounds"]

    return params_bounds


def define_fitness_calculator(
        protocols, feature_file, feature_set, probe=None, objective_weight_mea=2.5,
        interp_step=0.1,
        **extra_kwargs
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

    feature_definitions = get_feature_definitions(feature_file, feature_set)

    objectives = []
    efeatures = {}

    for protocol_name, locations in feature_definitions.items():

        efeatures[protocol_name] = []

        for location, features in locations.items():

            for efel_feature_name, meanstd in features.items():

                feature_name = f'{protocol_name}.{location}.{efel_feature_name}'

                if protocols[protocol_name].stimuli[0].step_delay > 0.:
                    stimulus = protocols[protocol_name].stimuli[0]
                else:
                    stimulus = protocols[protocol_name].stimuli[1]

                kwargs = {
                    'exp_mean': meanstd[0],
                    'exp_std': meanstd[1],
                    'stim_start': stimulus.step_delay,

                }

                if location != 'MEA':
                    kwargs['stimulus_current'] = stimulus.step_amplitude

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
                    
                    recording_names = '%s.%s.LFP' % (protocol_name, location)
                    somatic_recording_name = f'{protocol_name}.soma.v'

                    kwargs.update(extra_kwargs)

                    # depending on "exp_std" value, different strategies can be identified
                    if kwargs["exp_std"] is None:
                        # full cosine dist strategy
                        objective_weight = objective_weight_mea
                        channel_ids = None
                    elif np.isscalar(kwargs["exp_std"]):
                        # single channel strategy
                        objective_weight = 1  # in this case weight is the same as other intra-features
                        channel_ids = int(efel_feature_name.split("_")[-1])
                        efel_feature_name = "_".join(efel_feature_name.split("_")[:-1])
                    else:
                        # sections strategy
                        objective_weight = objective_weight_mea
                        channel_ids = kwargs["exp_std"]
                        efel_feature_name = "_".join(efel_feature_name.split("_")[:-1])
                        kwargs["exp_std"] = None

                    feature = ephys.efeatures.extraFELFeature(
                        name=feature_name,
                        extrafel_feature_name=efel_feature_name,
                        recording_names={'': recording_names},
                        somatic_recording_name=somatic_recording_name,
                        channel_ids=channel_ids,
                        interp_step=interp_step,
                        **kwargs
                    )

                else:

                    objective_weight = 1

                    recording_names = {'': '%s.%s.v' % (protocol_name, location)}

                    feature = ephys.efeatures.eFELFeature(
                        name=feature_name,
                        efel_feature_name=efel_feature_name,
                        recording_names=recording_names,
                        max_score=250,
                        force_max_score=True,
                        int_settings={'strict_stiminterval': True},
                        interp_step=interp_step,
                        **kwargs
                    )

                efeatures[protocol_name].append(feature)

                objectives.append(
                    ephys.objectives.SingletonWeightObjective(
                        feature_name,
                        feature,
                        weight=objective_weight
                    )
                )

    return ephys.objectivescalculators.ObjectivesCalculator(
        objectives), efeatures


def create_evaluator(
        model_name,
        cell_model_folder,
        feature_set,
        feature_file,
        protocol_file,
        probe_file=None,
        probe_type=None,
        protocols_with_lfp=None,
        extra_recordings=None,
        release=False,
        timeout=900.,
        morphology_file=None,
        v_init=None,
        abd=False,
        simulator="lfpy",
        **extra_kwargs
):
    """
    Prepares objects for optimization of the model.

    Parameters
    ----------
    model_name: str
        "hay", "hay_ais", or "hallermann"
    feature_set: str
        "soma", "extra"
    feature_file: str or Path
        Path to feature json file
    protocol_file: str or Path
        Path to feature json file
    probe_file: Path or None
        If given, the probe is loaded from the provided file (.json)
    probe_type: str or MEAutility.MEA
        If string, it can be "linear" or "planar", otherwise any MEAutility.MEA objects can be used
    protocols_with_lfp: list or None
        If given, the list of protocols to compute LFP from
    extra_recordings: dict or None
        If given, it specifies a set of extra recordings to add to a specific protocol. It needs to be in the form of:
        extra_recordings = {"protocol_name": [
                {
                "var": "v",
                "comp_x": 0.5,
                "type": "nrnseclistcomp",
                "name": "apical_middle",
                "seclist_name": "apical",
                "sec_index": 0
                },
        ]}
    timeout: float
        Timeout in seconds
    abd: bool
        If True and model is 'experimental', the ABD section is used
    simulator: str
        The simulator and cell models to use. "lfpy" | "neuron"
    extra_kwargs: keyword arguments for computing extracellular signals.

    Returns
    -------
    CellEvaluator
    """
    probe = None

    if model_name == 'experimental':
        assert morphology_file is not None, "Experimental model requires morphology file to be specified."
        cell = create_experimental_model(morphology_file, cell_model_folder, v_init=v_init, abd=abd,
                                         model_type=simulator)
    else:
        cell = create_ground_truth_model(model_name, cell_model_folder, release=release, v_init=v_init,
                                         model_type=simulator)

    if feature_set == "extra":
        if probe_file is not None:
            probe = define_electrode(probe_file=probe_file)
        elif probe_type is not None:
            probe = define_electrode(probe_type=probe_type)
        else:
            raise Exception(
                    "Probe must be provided for 'extra' feature set with"
                    "'probe_type' or 'probe_file' arguments"
                )

    param_names = [param.name for param in cell.params.values() if not param.frozen]

    fitness_protocols = define_protocols(
        model_name,
        feature_set=feature_set,
        feature_file=feature_file,
        protocols_file=protocol_file,
        electrode=probe,
        protocols_with_lfp=protocols_with_lfp,
        extra_recordings=extra_recordings,
        simulator=simulator
    )

    fitness_calculator, _ = define_fitness_calculator(
        protocols=fitness_protocols,
        feature_file=feature_file,
        feature_set=feature_set,
        probe=probe,
        **extra_kwargs
    )

    if simulator == "lfpy":
        sim = ephys.simulators.LFPySimulator(cell, cvode_active=True, electrode=probe,
                                             mechs_folders=cell_model_folder)
    else:
        sim = ephys.simulators.NrnSimulator(dt=None, cvode_active=True,
                                            mechs_folders=cell_model_folder)

    return ephys.evaluators.CellEvaluator(
        cell_model=cell,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=sim,
        timeout=timeout
    )