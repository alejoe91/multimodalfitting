import numpy as np
import neo
from pathlib import Path
import json
from copy import deepcopy

from bluepyefe.reader import _check_metadata
from bluepyopt.ephys.extra_features_utils import calculate_features, all_1D_features

from ..ecode import HyperDepol, sAHP, PosCheops
from .targets import get_firepattern_targets, get_idrest_targets, get_iv_targets, get_apwaveform_targets, \
    get_hyperdepol_targets, get_sahp_targets, get_poscheops_targets


protocol_name_to_target_function = {
    "firepattern": get_firepattern_targets,
    "idrest": get_idrest_targets,
    "iv": get_iv_targets,
    "apwaveform": get_apwaveform_targets,
    "hyperdepol": get_hyperdepol_targets,
    "sahp": get_sahp_targets,
    "poscheops": get_poscheops_targets
}


###### BUILD METADATA #####
def get_ecode_protocol_names():
    return ecodes_wcp_timings.keys()


def build_wcp_metadata(cell_id, files_list, ecode_timings, repetition_as_different_cells=False,
                       liquid_junction_potential=14.):
    """
    Builds metadata for experimental data in wcp format.

    Parameters
    ----------
    cell_id: str
        Name of the cell (arbutrary)
    files_list: list if dict
        List (one element for each run) of dictionaries indicating the files associated to different protocols:
        Example:
        files_list = [
            # run1
            {"firepattern": path-to-firepattern-run1.wcp
            ...
            },
            # run2
            {"firepattern": path-to-firepattern-run2.wcp
            ...
            },
        ]
    ecode_timings: dict
        Dictionary with timings associated to the different ecode protocols
    repetition_as_different_cells: bool
        Whether different repetitions should be considered as different cells (False by default)
    liquid_junction_potential: float
        The liquid juncion potential for the experiment (default 14 mV)

    Returns
    -------
    files_metadata: dict
        Dictionary with extracted metadata

    """

    files_metadata = {}

    if not repetition_as_different_cells:
        files_metadata[cell_id] = {}

    for i_rep, repetition_dict in enumerate(files_list):
        if repetition_as_different_cells:
            current_id = f"{cell_id}_rep{i_rep}"
            files_metadata[current_id] = {}
        else:
            current_id = cell_id

        for ecode_protocol in repetition_dict:
            file_path = repetition_dict[ecode_protocol]
            # file_path = Path(ephys_dir) / f"{cell_id}_run{repetition}.{ecode_to_index[ecode]}.wcp"
            # print(file_path.name)

            if not file_path.is_file():
                print(f"Missing trace {file_path}")
                continue

            metadata = {
                "filepath": str(file_path),
                "i_unit": "pA",
                "t_unit": "s",
                "v_unit": "mV",
                "ljp": liquid_junction_potential
            }

            metadata.update(ecode_timings[ecode_protocol])

            if ecode_protocol not in files_metadata[current_id]:
                files_metadata[current_id][ecode_protocol] = [metadata]
            else:
                files_metadata[current_id][ecode_protocol].append(metadata)

    return files_metadata


def build_model_metadata(cell_id, ephys_dir):
    files_metadata = {}

    files_metadata[cell_id] = {}

    ephys_dir = Path(ephys_dir)

    for protocol_folder in ephys_dir.iterdir():
        ecode = protocol_folder.name
        if "extracellular" not in protocol_folder.name and not protocol_folder.name.startswith(".")\
                and "efeatures" not in protocol_folder.name and "results" not in protocol_folder.name:
            files_metadata[cell_id][ecode] = []
            for sweep in protocol_folder.iterdir():
                metadata = {
                    "filepath": str(sweep),
                    "i_unit": "nA",
                    "t_unit": "ms",
                    "v_unit": "mV",
                }
                metadata.update(ecodes_model_timings[ecode])
                files_metadata[cell_id][ecode].append(metadata)

    return files_metadata


#### Ecode params ###

ecode_to_index = {
    "IDthres": 0,
    "firepattern": 1,
    "IV": 2,
    "IDrest": 3,
    "APWaveform": 4,
    "HyperDepol": 5,
    "sAHP": 6,
    "PosCheops": 7
}

ecodes_wcp_timings = {
    "IDthres": {
        'ton': 200,
        'toff': 470
    },
    "firepattern": {
        'ton': 500,
        'toff': 4100
    },
    "IV": {
        'ton': 250,
        'toff': 3250
    },
    "IDrest": {
        'ton': 200,
        'toff': 1550
    },
    "APWaveform": {
        'ton': 150,
        'toff': 200
    },
    "HyperDepol": {
        'ton': 200,
        'toff': 920,
        'tmid': 650
    },
    "sAHP": {
        'ton': 200,
        'toff': 1125,
        'tmid': 450,
        'tmid2': 675
    },
    "PosCheops": {
        'ton': 1000,
        't1': 9000,
        't2': 10500,
        't3': 14500,
        't4': 16000,
        'toff': 18660
    }
}

ecodes_model_timings = {
    "IDthres": {
        'ton': 250,
        'toff': 520
    },
    "firepattern": {
        'ton': 250,
        'toff': 3850
    },
    "IV": {
        'ton': 250,
        'toff': 3250
    },
    "IDrest": {
        'ton': 250,
        'toff': 1600
    },
    "APWaveform": {
        'ton': 250,
        'toff': 300
    },
    "HyperDepol": {
        'ton': 250,
        'toff': 970,
        'tmid': 700
    },
    "sAHP": {
        'ton': 250,
        'toff': 1175,
        'tmid': 500,
        'tmid2': 725
    },
    "PosCheops": {
        'ton': 250,
        't1': 8250,
        't2': 9750,
        't3': 13750,
        't4': 15250,
        'toff': 17910
    }
}


###### READERS ########
def wcp_reader(in_data, voltage_channel=0, current_channel=0):
    """Reader for .wcp

    Args:
        in_data (dict): of the format
        {
            "filepath": "./XXX.wcp",
            "i_unit": "pA",
            "t_unit": "s",
            "v_unit": "mV",
        }
    """

    _check_metadata(
        in_data,
        wcp_reader.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    # Read file
    io = neo.WinWcpIO(in_data["filepath"])
    block = io.read_block()

    data = []
    for segment in block.segments:

        voltage_array = segment.analogsignals[0].as_array()
        current_array = segment.analogsignals[1].as_array()
        sr = segment.analogsignals[0].sampling_rate
        dt = 1.0 / sr.magnitude

        if voltage_array.shape[1] > 1:
            voltage_trace = voltage_array[:, voltage_channel]
        else:
            voltage_trace = np.squeeze(voltage_array)
        if current_array.shape[1] > 1:
            current_trace = current_array[:, current_channel]
        else:
            current_trace = np.squeeze(current_array)

        trace_data = {
            "voltage": voltage_trace,
            "current": current_trace,
            "dt": dt
        }

        data.append(trace_data)

    return data


def model_csv_reader(in_data):
    """Reader for modeled csv data

    Args:
        in_data (dict): of the format
        {
            "folderpath": "./XXX",
            "i_unit": "nA",
            "t_unit": "s",
            "v_unit": "mV",
        }
    """
    import pandas as pd

    _check_metadata(
        in_data,
        model_csv_reader.__name__,
        ["filepath", "i_unit", "v_unit", "t_unit"],
    )

    # Read file
    data = []
    sweep_df = pd.read_csv(in_data["filepath"])
    trace_data = {
        "voltage": np.array(sweep_df["voltage"]),
        "current": np.array(sweep_df["current"]),
        "dt": np.median(np.diff(sweep_df["time"]))
    }

    data.append(trace_data)

    return data

##### TARGETS ########
def get_ecode_targets(timings, include_pre_post=True):
    targets = []
    for protocol_name, target_function in protocol_name_to_target_function.items():
        new_targets = target_function(timings=timings, include_pre_post=include_pre_post)
        targets += new_targets
    return targets


###### CONVERT TO BPO #####
def convert_to_bpo_format(in_protocol_path, in_efeatures_path,
                          out_protocol_path=None, out_efeatures_path=None,
                          epsilon=1e-3, protocols_of_interest=None,
                          exclude_features=None,
                          std_from_mean=None):
    """
    Converts protocols and features from BluePyEfe to BPO format.

    Parameters
    ----------
    in_protocol_path: str or Path
        Path to input protocols json file
    in_efeatures_path: str or Path
        Path to input efeature json file
    out_protocol_path: str or Path
        Path to output protocols json file
    out_efeatures_path: str or Path
        Path to output efeatures json file
    epsilon: float
        Value to substitute to features with 0 std (default 1e-3)
    protocols_of_interest: list or None
        If not None, list of protocols to export
    exclude_features: dict
        Dictionary with efeatures to exclude from single protocols
    std_from_mean: float or None
        If not None, the std of features is std_from_mean times the mean

    Returns
    -------
    protocols_dict: dict
        Dictionary with protocol dict saved to json
    efeatures_dict: dict
        Dictionary with efeatures dict saved to json

    """
    in_protocols = json.load(open(in_protocol_path, 'r'))
    in_efeatures = json.load(open(in_efeatures_path, 'r'))

    out_efeatures = {}

    # feature_set = "soma"
    # out_efeatures[feature_set] = {}

    out_protocols = {}
    out_efeatures = {}

    if protocols_of_interest is None:
        protocols_of_interest = list(in_protocols.keys())

    print(f"Saving protocols and features for the following protocols:\n{protocols_of_interest}")

    for protocol_name in protocols_of_interest:
        if protocol_name in in_protocols and protocol_name in in_efeatures:

            # Convert the format of the protocols
            stimuli = [
                in_protocols[protocol_name]['holding'],
                in_protocols[protocol_name]['step']
            ]
            out_protocols[protocol_name] = {'stimuli': stimuli}

            # Convert the format of the efeatures
            out_efeatures[protocol_name] = {}
            for loc_name, features in in_efeatures[protocol_name].items():
                efeatures_list = []

                for feature in features:
                    add_feature = True
                    if exclude_features is not None:
                        if protocol_name in exclude_features:
                            if feature["feature"] in exclude_features[protocol_name]:
                                add_feature = False
                    if add_feature:
                        efeatures_dict = feature
                        if std_from_mean is not None:
                            efeatures_dict["val"][1] = np.abs(std_from_mean * efeatures_dict["val"][0])
                        if efeatures_dict["val"][1] == 0:
                            efeatures_dict["val"][1] = epsilon
                    else:
                        print(f"Excluding efeature {feature['feature']} from protocol {protocol_name}")
                    efeatures_list.append(efeatures_dict)
                out_efeatures[protocol_name][loc_name] = efeatures_list

    if out_protocol_path is not None:
        s = json.dumps(out_protocols, indent=2)
        with open(out_protocol_path, "w") as fp:
            fp.write(s)

    if out_efeatures_path is not None:
        s = json.dumps(out_efeatures, indent=2)
        with open(out_efeatures_path, "w") as fp:
            fp.write(s)

    return out_protocols, out_efeatures


def append_extrafeatures_to_json(extra_features, protocol_name, efeatures_dict, efeatures_path=None, channel_ids=None,
                                 single_channel_features=False, std_from_mean=None, epsilon=1e-3):
    """
    Appends extracellualar features to existing efeatures json file in BPO format

    Parameters
    ----------
    extra_features: dict
        Dictionary with extracellular features (computed with compute_extra_features()) function)
    protocol_name: str
        Protocol name to compute extracellular features in optimization. It should be a protocol that elicits
        a large enough number of spikes
    efeatures_dict: dict
        Dictionary with intracellular efeatures
    efeatures_path: str or Path or None
        Path to input efeature json file in BPO format to modify in place. If None, the output is not saved
    channel_ids: list, lists, or None
        - If None, features from all channels are saved in the json file. Std is set to None.
        - If list and 'single_channel_features' is True, features from different channels are saved as separate
        features and the channel id is appended to the feature name. If std_from_mean is given, the second element
        of each feature list (the standard deviation) is computed from the mean
        - If list of lists, several features for each extra feature are saved, corresponding to different sections
        of the MEA (each element is a list of channels that makes a section). In this case, the channel_ids for each
        section are also saved in the second element of the feature
    single_channel_features: bool
        If True and 'channel_ids' is a list, features from different channels are saved as separate features
    std_from_mean: float or None
        If 'single_channel_features' is True, the std of the feature is std_from_mean times the feature mean
    epsilon: float
        If 'single_channel_features' and the feature mean is 0, the value of the std is set to epsilon (default 1e-5)


    Returns
    -------
    efeatures_dict: dict
       Modified dictionary with efeatures dict, including MEA features, saved to json

    """
    # TODO refactor to new list-based format
    new_efeatures_dict = deepcopy(efeatures_dict)

    if protocol_name not in list(new_efeatures_dict.keys()):
        # create protocol if non existing
        new_efeatures_dict[protocol_name] = {}

    # create MEA location
    append_features = []
    for extra_feat_name, feature_values in extra_features.items():
        feature_list = feature_values.tolist()
        feature_dict = {"n": 1,
                        "efel_settings": {}}
        if channel_ids is None:
            feature_dict["feature"] = extra_feat_name
            feature_dict["val"] = [feature_list, None]
            append_features.append(deepcopy(feature_dict))
        elif isinstance(channel_ids, list) and np.isscalar(channel_ids[0]) and single_channel_features:
            # save channels separately
            assert std_from_mean is not None, "When 'single_channel_features' is used, the 'std_from_mean' argument " \
                                              "should be specified"
            for chan in channel_ids:
                std = np.abs(std_from_mean * feature_list[chan])
                if std == 0:
                    print(f"Setting {extra_feat_name} channel {chan} std to {epsilon}")
                    std = epsilon
                feature_dict["feature"] = f"{extra_feat_name}_{chan}"
                feature_dict["val"] = [feature_list[chan], std]
                append_features.append(deepcopy(feature_dict))
        else:
            if np.isscalar(channel_ids[0]):
                # subset
                feature_dict["feature"] = extra_feat_name
                feature_dict["val"] = [list(np.array(feature_list)[channel_ids]), channel_ids]
                append_features.append(deepcopy(feature_dict))
            else:
                for sec, channels in enumerate(channel_ids):
                    feature_dict["feature"] = f"{extra_feat_name}_{sec}"
                    feature_dict["val"] = [list(np.array(feature_list)[channels]), channels]
                    append_features.append(deepcopy(feature_dict))
    new_efeatures_dict[protocol_name]["MEA"] = append_features

    if efeatures_path is not None:
        s = json.dumps(new_efeatures_dict, indent=2)
        with open(efeatures_path, "w") as fp:
            fp.write(s)

    return new_efeatures_dict


def compute_extra_features(eap, fs, upsample=1, feature_list=None):
    """
    Computes extracellular feature arrays from extracellular action potential.

    Parameters
    ----------
    eap: np.array (num_channels x num_samples)
        The extracellular action potential
    fs: float
        Sampling frequency in Hz
    upsample: int or None
        If given, the template (eap) is upsampled by 'upsample' times before computing features
    feature_list: list or None
        List of extracellular features to compute. If None, all available features are computed.

    Returns
    -------
    features: dict
        Dictionary with computed features
    """
    if feature_list is None:
        feature_list = all_1D_features
    else:
        for feat_name in feature_list:
            assert feat_name in all_1D_features, f"{feat_name} is not in available feature list {all_1D_features}"

    features = calculate_features(eap, sampling_frequency=fs, upsample=upsample,
                                  feature_names=feature_list)

    return features
