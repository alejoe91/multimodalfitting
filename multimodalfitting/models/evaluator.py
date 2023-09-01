"""Run simple cell optimisation"""

import json
import pickle
import pathlib
from pathlib import Path
import numpy as np

import bluepyopt.ephys as ephys

import logging

from .model import define_electrode, create_ground_truth_model, create_experimental_model
# from ..efeatures_extraction import get_sahp_efeatures, get_hyper_depol_efeatures, get_poscheops_efeatures

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
                    if feature['feature'] in efeatures_def:
                        orig_feature_name = feature['feature']
                        feature_name = orig_feature_name
                        i = 1
                        while feature_name in efeatures_def:
                            feature_name = f"{orig_feature_name}_{i}"
                            i += 1
                        efel_feature_name = orig_feature_name
                    else:
                        efel_feature_name = feature['feature']
                        feature_name = feature['feature']
                    efeatures_def[feature_name] = {}
                    efeatures_def[feature_name]['mean'] = feature['val'][0]
                    efeatures_def[feature_name]['std'] = feature['val'][1]

                    if std_from_mean is not None:
                        efeatures_def[feature_name]['std'] = np.abs(
                            std_from_mean * efeatures_def[feature_name]['mean'])
                    if efeatures_def[feature_name]['std'] == 0:
                        efeatures_def[feature_name]['std'] = epsilon
                    efeatures_def[feature_name]['efel_settings'] = feature['efel_settings']
                    efeatures_def[feature_name]['efel_feature_name'] = efel_feature_name
                else:
                    print(f"Excluding efeature {feature_name} from protocol {protocol_name}")
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


def get_feature_definitions(feature_file):
    """
    Returns features definitions

    Parameters
    ----------
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
                if var == "v":
                    recording_class = ephys.recordings.CompRecording
                elif var == "i_membrane":
                    from ..bpopt_ext import CompCurrentRecording
                    recording_class = CompCurrentRecording
                recordings.append(
                    recording_class(
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


def define_stimuli(protocol_definition, simulator="lfpy",
                   protocol_name=None):
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
    if protocol_name is None:
        protocol_name = "step"

    if simulator.lower() == "lfpy":
        stimulus_step_class = ephys.stimuli.LFPySquarePulse
    else:
        stimulus_step_class = ephys.stimuli.NrnSquarePulse

    if any(stim in protocol_name.lower() for stim in ["sahp", "hyperdepol", "poscheops"]):
        from ..ecode import HyperDepol, sAHP, PosCheops
        stimulus_holding = protocol_definition["stimuli"][0]
        stimuli.append(stimulus_step_class(
            step_amplitude=stimulus_holding['amp'],
            step_delay=stimulus_holding['delay'],
            step_duration=stimulus_holding['duration'],
            location=soma_loc,
            total_duration=stimulus_holding['totduration']))
        stimulus_definition = protocol_definition["stimuli"][1]
        if "sahp" in protocol_name.lower():
            sahp = sAHP(delay=stimulus_definition["delay"],
                        tmid=stimulus_definition["tmid"],
                        tmid2=stimulus_definition["tmid2"],
                        toff=stimulus_definition["toff"],
                        total_duration=stimulus_definition["totduration"],
                        phase1_amplitude=stimulus_definition["long_amp"],
                        phase2_amplitude=stimulus_definition["amp"],
                        phase3_amplitude=stimulus_definition["long_amp"],
                        location=soma_loc)
            stimuli.append(sahp)
        elif "hyperdepol" in protocol_name.lower():
            hyperdepol = HyperDepol(delay=stimulus_definition["delay"],
                                    tmid=stimulus_definition["tmid"],
                                    toff=stimulus_definition["toff"],
                                    total_duration=stimulus_definition["totduration"],
                                    hyperpol_amplitude=stimulus_definition["amp"],
                                    depol_amplitude=stimulus_definition["amp2"],
                                    location=soma_loc)
            stimuli.append(hyperdepol)
        elif "poscheops" in protocol_name.lower():
            poscheops = PosCheops(delay=stimulus_definition["delay"],
                                  t1=stimulus_definition["t1"],
                                  t2=stimulus_definition["t2"],
                                  t3=stimulus_definition["t3"],
                                  t4=stimulus_definition["t4"],
                                  toff=stimulus_definition["toff"],
                                  total_duration=stimulus_definition["totduration"],
                                  ramp1_amp=stimulus_definition["amp"],
                                  ramp2_amp=stimulus_definition["amp"],
                                  ramp3_amp=stimulus_definition["amp"],
                                  location=soma_loc)
            stimuli.append(poscheops)
    else:
        for stimulus_definition in protocol_definition["stimuli"]:
            stimuli.append(stimulus_step_class(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=soma_loc,
                total_duration=stimulus_definition['totduration']))

    return stimuli


def define_protocols(
        model_name,
        feature_file=None,
        protocols_file=None,
        electrode=None,
        protocols_with_lfp=None,
        extra_recordings=None,
        simulator="lfpy",
        all_protocols=False,
        exclude_protocols=None
):
    """
    Defines protocols for a specified feature_Set (or file)

    Parameters
    ----------
    model_name: str
        "hay", "hallermann", "cultured"
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
    feature_definitions = get_feature_definitions(feature_file)

    if all_protocols:
        protocol_definitions = convert_all_protocols(protocol_definitions)
        # feature_definitions = convert_all_features(
        #     feature_definitions, protocol_definitions)

    assert simulator.lower() in ["lfpy", "neuron"]

    protocols = {}

    if protocols_with_lfp is None:
        protocols_with_lfp = []
    if exclude_protocols is None:
        exclude_protocols = []

    for protocol_name in feature_definitions:
        if np.all([excl not in protocol_name for excl in exclude_protocols]):
            if protocol_name in protocols_with_lfp:
                recordings = define_recordings(
                    protocol_name, protocol_definitions[protocol_name], electrode, extra_recordings
                )
            else:
                recordings = define_recordings(
                    protocol_name, protocol_definitions[protocol_name], None, extra_recordings
                )

            stimuli = define_stimuli(protocol_definitions[protocol_name], simulator=simulator,
                                     protocol_name=protocol_name)

            protocols[protocol_name] = ephys.protocols.SweepProtocol(
                protocol_name, stimuli, recordings, cvode_active=True
            )

    return protocols


def define_test_step_protocol(step_amplitude=0.5, tot_duration=500, delay=50,
                              step_duration=400, probe=None, protocol_name="TestStep",
                              simulator="lfpy", extra_recordings=None):
    """Generates tests protocol with a current pulse.

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
        protocols, feature_file, probe=None, objective_weight_mea=2.5,
        interp_step=0.1, exclude_protocols=None,
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
    probe: MEAutility.MEA
        The probe to use for extracellular features
    objective_weight_mea: float
        Weight for cosine distance ectraFELFeatures (default=2.5)
    interp_step: float
        EFEl interpolation step (default=0.1)
    exclude_protocols: list or None
        If given, list of protocols to exclude from fitness objectives    
    **extra_kwargs: keyword arguments for extracellular fitness (see `multimodalfitting.get_extra_kwargs()`)

    Returns
    -------
    objectives_calculator
    """
    # if feature_set is not None:
    #     assert feature_set in ['multiple', 'soma', 'extra'], "feature_set should be 'multiple', 'soma' or 'extra'."

    #     if feature_set == 'extra' and probe is None:
    #         raise Exception("Provide a MEAutility probe to use the 'extra' set.")

    #     feature_definitions = get_feature_definitions(feature_file, feature_set)
    # else:
    # feature_definitions = get_feature_definitions(
    #     feature_file, None)
    # feature_definitions = convert_all_features(feature_definitions, protocols)
    feature_definitions = get_feature_definitions(feature_file)

    objectives = []
    efeatures = {}

    if exclude_protocols is None:
        exclude_protocols = []
    for protocol_name, locations in feature_definitions.items():
        if np.all([excl not in protocol_name for excl in exclude_protocols]):
            efeatures[protocol_name] = []
            for location, features in locations.items():
                for feat in features:
                    if "val" not in feat:
                        efeat_values = features[feat]
                        feat_name = feat
                        if isinstance(efeat_values, dict):
                            mean = efeat_values['mean']
                            std = efeat_values['std']
                            efel_settings = efeat_values['efel_settings']
                            efel_feature_name = efeat_values.get(
                                'efel_feature_name', feat_name)
                        else:
                            mean = efeat_values[0]
                            std = efeat_values[1]
                            efel_settings = {}
                            efel_feature_name = feat_name
                    else: # new format
                        feat_name = feat["feature"]
                        mean = feat["val"][0]
                        std = feat["val"][1]
                        efel_settings = feat['efel_settings']
                        efel_feature_name = feat.get(
                            'efel_feature_name', feat_name)
                    feature_name = f'{protocol_name}.{location}.{feat_name}'

                    if protocols[protocol_name].stimuli[0].step_delay > 0.:
                        stimulus = protocols[protocol_name].stimuli[0]
                    else:
                        stimulus = protocols[protocol_name].stimuli[1]

                    kwargs = {
                        'exp_mean': mean,
                        'exp_std': std,
                    }

                    stim_kwargs = {}

                    if isinstance(stimulus, (ephys.stimuli.NrnSquarePulse, ephys.stimuli.LFPySquarePulse)):
                        stim_kwargs['stim_start'] = efel_settings.get(
                            'stim_start', stimulus.step_delay)
                        stim_kwargs['stim_end'] = efel_settings.get(
                            'stim_end', stimulus.step_delay + stimulus.step_duration)
                        stim_kwargs['stimulus_current'] = efel_settings.get(
                            'stimulus_current', stimulus.step_amplitude)
                    else:
                        stim_kwargs = efel_settings

                    feature, objective = _get_feature_and_objective(feature_name, efel_feature_name, protocol_name,
                                                                    location, objective_weight_mea, stim_kwargs,
                                                                    extra_kwargs, kwargs, interp_step)

                    objectives.append(objective)
                    efeatures[protocol_name].append(feature)

    return ephys.objectivescalculators.ObjectivesCalculator(
        objectives), efeatures


def _get_feature_and_objective(feature_name, efel_feature_name, protocol_name, location, objective_weight_mea,
                               stim_kwargs, extra_kwargs, kwargs, interp_step):
    if location == 'MEA':
        del stim_kwargs['stimulus_current']

    kwargs.update(stim_kwargs)

    if location == 'soma':
        kwargs['threshold'] = -20
    elif 'dend' in location:
        kwargs['threshold'] = -55
    else:
        kwargs['threshold'] = -20

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
            max_score=250,
            force_max_score=True,
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
    objective = ephys.objectives.SingletonWeightObjective(
        feature_name,
        feature,
        weight=objective_weight
    )
    return feature, objective


def create_evaluator(
        model_name,
        strategy,
        protocols_with_lfp=None,
        extra_recordings=None,
        cell_folder=None,
        release=False,
        timeout=900.,
        abd=False,
        extracellularmech=False,
        optimize_ra=False,
        simulator="lfpy",
        interp_step=0.1,
        all_protocols=False,
        exclude_protocols=None,
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
    strategy: str
        "soma", "all", "single", "sections", ("validation")
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
    extracellularmech: bool
        If True, extracellular mechanism is inserted into the model for recording i_membrane
        Default is False
    optimize_ra: bool
        If True and abd is True, Ra is also oprimized for AIS and ABD
    simulator: str
        The simulator and cell models to use. "lfpy" | "neuron"
    all_protocols: bool
        If True, all protocols (and features) are used
    exclude_protocols: list or None
        List of protocols (and associated features) to exclude from evaluator
    extra_kwargs: keyword arguments for computing extracellular signals.

    Returns
    -------
    CellEvaluator
    """
    if extra_recordings is not None:
        if not extracellularmech:
            print("Setting 'extracellularmech' to True for extra_recordings")
            extracellularmech = True
    probe = None
    if strategy:
        assert strategy in ["soma", "all", "sections", "single", "validation"]
    if model_name not in ['hay', 'hay_ais', 'hay_ais_hillock']:
        cell = create_experimental_model(model_name=model_name, abd=abd, optimize_ra=optimize_ra, 
                                         model_type=simulator, extracellularmech=extracellularmech)
    else:
        cell = create_ground_truth_model(model_name=model_name, release=release, model_type=simulator,
                                         extracellularmech=extracellularmech)

    if cell_folder is None:
        cell_folder = cell_models_folder
    cell_model_folder = cell_folder / model_name
    fitting_folder = cell_model_folder / "fitting"
    efeatures_folder = fitting_folder / "efeatures"

    assert efeatures_folder.is_dir(), f"Couldn't find fitting folder {efeatures_folder}"

    if strategy in ["all", "sections", "single", "validation"]:
        probe_file = efeatures_folder / "probe_BPO.json"
        assert probe_file.is_file() is not None, f"Couldn't find probe file {probe_file}"
        probe = define_electrode(probe_file=probe_file)

    if not all_protocols:
        features_file = efeatures_folder / f"features_BPO_{strategy}.json"
        protocols_file = efeatures_folder / f"protocols_BPO_{strategy}.json"
    else:
        features_file = efeatures_folder / "features.json"
        protocols_file = efeatures_folder / "protocols.json"

    assert features_file.is_file() is not None, f"Couldn't find features file {features_file}"
    assert protocols_file.is_file() is not None, f"Couldn't find protocols file {protocols_file}"

    param_names = [param.name for param in cell.params.values() if not param.frozen]

    fitness_protocols = define_protocols(
        model_name,
        feature_file=features_file,
        protocols_file=protocols_file,
        electrode=probe,
        protocols_with_lfp=protocols_with_lfp,
        extra_recordings=extra_recordings,
        simulator=simulator,
        all_protocols=all_protocols,
        exclude_protocols=exclude_protocols
    )

    fitness_calculator, _ = define_fitness_calculator(
        protocols=fitness_protocols,
        feature_file=features_file,
        probe=probe,
        interp_step=interp_step,
        exclude_protocols=exclude_protocols,
        **extra_kwargs
    )

    if simulator.lower() == "lfpy":
        sim = ephys.simulators.LFPySimulator(cell, cvode_active=True, electrode=probe,
                                             mechanisms_directory=cell_model_folder)
    else:
        sim = ephys.simulators.NrnSimulator(dt=None, cvode_active=True,
                                            mechanisms_directory=cell_model_folder)

    return ephys.evaluators.CellEvaluator(
        cell_model=cell,
        param_names=param_names,
        fitness_protocols=fitness_protocols,
        fitness_calculator=fitness_calculator,
        sim=sim,
        timeout=timeout
    )
