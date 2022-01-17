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

this_file = Path(__file__)
cell_models_folder = this_file.parent.parent.parent / "cell_models"


def convert_all_protocols(protocols_dict, protocols_of_interest=None):
    in_protocols = protocols_dict
    out_protocols = {}

    if protocols_of_interest is None:
        protocols_of_interest = list(in_protocols.keys())

    for protocol_name in protocols_of_interest:
        # if protocol_name in in_protocols and protocol_name in in_efeatures:

        # Convert the format of the protocols
        stimuli = [
            in_protocols[protocol_name]['holding'],
            in_protocols[protocol_name]['step']
        ]
        out_protocols[protocol_name] = {'stimuli': stimuli}
    
    return out_protocols


def convert_all_features(features_dict, protocols_dict, std_from_mean=0.05,
                         epsilon=1e-3, exclude_features=None):
    in_efeatures = features_dict
    out_efeatures = {}

    for protocol_name in protocols_dict:
        # if protocol_name in in_protocols and protocol_name in in_efeatures:

        # Convert the format of the efeatures
        efeatures_def = {}
        for loc_name, features in in_efeatures[protocol_name].items():
            for feature in features:
                add_feature = True
                if exclude_features is not None:
                    if protocol_name in exclude_features:
                        if feature["feature"] in exclude_features[protocol_name]:
                            add_feature = False
                if add_feature:
                    efeatures_def[feature['feature']] = feature['val']
                    if std_from_mean is not None:
                        efeatures_def[feature['feature']][1] = np.abs(std_from_mean *
                                                                      efeatures_def[feature['feature']][0])
                    if efeatures_def[feature['feature']][1] == 0:
                        efeatures_def[feature['feature']][1] = epsilon
                else:
                    print(f"Excluding efeature {feature['feature']} from protocol {protocol_name}")
            out_efeatures[protocol_name] = {loc_name: efeatures_def}

    return out_efeatures


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

    if simulator.lower() == "lfpy":
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
        simulator="lfpy", 
        all_protocols=False
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
    
    if all_protocols:
        protocol_definitions = convert_all_protocols(protocol_definitions)
        feature_definitions = convert_all_features(
            feature_definitions, protocol_definitions)

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
                              simulator="lfpy", extra_recordings=None):
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
    recordings = define_recordings(protocol_name, protocol_definition, probe, extra_recordings=extra_recordings)

    stimuli = define_stimuli(protocol_definition, simulator=simulator)

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
    if feature_set is not None:
        assert feature_set in ['multiple', 'soma', 'extra'], "feature_set should be 'multiple', 'soma' or 'extra'."

        if feature_set == 'extra' and probe is None:
            raise Exception("Provide a MEAutility probe to use the 'extra' set.")

        feature_definitions = get_feature_definitions(feature_file, feature_set)
    else:
        feature_definitions = get_feature_definitions(
            feature_file, None)
        feature_definitions = convert_all_features(feature_definitions, protocols)

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
        feature_set,
        extra_strategy=None,
        protocols_with_lfp=None,
        extra_recordings=None,
        cell_folder=None,
        release=False,
        timeout=900.,
        abd=False,
        optimize_ra=False,
        simulator="lfpy",
        interp_step=0.1,
        all_protocols=False,
        **extra_kwargs
):
    """
    Prepares objects for optimization of the model.
    The cell model folder ("multimodal_fitting/cell_models/{model_name}") needs to have the following files:
    * a morphology file (swc or asc with "morphology" in the file name)
    * mechanisms.json
    * parameters.json and parameters_release.json (for experimental models also abd and abd_ra versions)
    * a mechanism folder with the mod files
    * a fitting folder with:
        * probe_BPO.json (description of the probe)
        * protocols_BPO_{}.json (protocol files for 'all', 'sections', and 'single' extra strategies)
        * features_BPO_{}.json (feature files for 'all', 'sections', and 'single' extra strategies)
        * holding_threshold_currents.json (information about hilding current)
        * features.json and protocols.json for "all_protocols" option

    Parameters
    ----------
    model_name: str
        "hay", "hay_ais", or "hallermann"
    feature_set: str
        "soma", "extra"
    extra_strategy: str or None
        "all", "sections", "single" (only needed if feature_set is "extra")
    protocols_with_lfp: list or None
        If given, the list of protocols to compute LFP from
    cell_folder: path or None
        If given, the cell_folder where the model_name folder is. Default is "multimodal/cell_folders"
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
    optimize_ra: bool
        If True and abd is True, Ra is also oprimized for AIS and ABD
    simulator: str
        The simulator and cell models to use. "lfpy" | "neuron"
    all_protocols: bool 
        If True, all protocols (and features) are used
    extra_kwargs: keyword arguments for computing extracellular signals.

    Returns
    -------
    CellEvaluator
    """
    probe = None
    if extra_strategy:
        assert extra_strategy in ["all", "sections", "single"]
    if model_name not in ['hay', 'hay_ais', 'hay_ais_hillock']:
        cell = create_experimental_model(model_name=model_name, abd=abd, optimize_ra=optimize_ra, model_type=simulator)
    else:
        cell = create_ground_truth_model(model_name=model_name, release=release, model_type=simulator)

    if cell_folder is None:
        cell_folder = cell_models_folder
    cell_model_folder = cell_folder / model_name
    fitting_folder = cell_model_folder / "fitting"
    efeatures_folder = fitting_folder / "efeatures"

    assert efeatures_folder.is_dir(), f"Couldn't find fitting folder {efeatures_folder}"

    if feature_set == "extra":
        probe_file = efeatures_folder / "probe_BPO.json"
        assert probe_file.is_file() is not None, f"Couldn't find probe file {probe_file}"
        probe = define_electrode(probe_file=probe_file)

        assert extra_strategy is not None, "'extra_strategy' must be specified for 'extra' feature_set"
    else:
        extra_strategy = "all"

    if not all_protocols:
        features_file = efeatures_folder / f"features_BPO_{extra_strategy}.json"
        protocols_file = efeatures_folder / f"protocols_BPO_{extra_strategy}.json"
    else:
        features_file = efeatures_folder / f"features.json"
        protocols_file = efeatures_folder / f"protocols.json"
        feature_set = None

    assert features_file.is_file() is not None, f"Couldn't find features file {features_file}"
    assert protocols_file.is_file() is not None, f"Couldn't find protocols file {protocols_file}"

    param_names = [param.name for param in cell.params.values() if not param.frozen]

    fitness_protocols = define_protocols(
        model_name,
        feature_set=feature_set,
        feature_file=features_file,
        protocols_file=protocols_file,
        electrode=probe,
        protocols_with_lfp=protocols_with_lfp,
        extra_recordings=extra_recordings,
        simulator=simulator,
        all_protocols=all_protocols
    )

    fitness_calculator, _ = define_fitness_calculator(
        protocols=fitness_protocols,
        feature_file=features_file,
        feature_set=feature_set,
        probe=probe,
        interp_step=interp_step,
        **extra_kwargs
    )

    if simulator.lower() == "lfpy":
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
