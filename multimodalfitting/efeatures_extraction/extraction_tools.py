import numpy as np
import neo
from pathlib import Path
import json
from copy import deepcopy

from bluepyefe.reader import _check_metadata
from bluepyopt.ephys.extra_features_utils import calculate_features, all_1D_features


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

def wcp_reader(in_data):
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

        trace_data = {
            "voltage": np.array(segment.analogsignals[0]).flatten(),
            "current": np.array(segment.analogsignals[1]).flatten(),
            "dt": 1.0 / int(segment.analogsignals[0].sampling_rate)
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

def _get_hyper_depol_efeatures(timings):
    hyper_depol_efeatures = {
        'Spikecount': {"stim_start": timings["HyperDepol"]["tmid"],
                       "stim_end": timings["HyperDepol"]["toff"]},
        'burst_number': {"stim_start": timings["HyperDepol"]["tmid"],
                         "stim_end": timings["HyperDepol"]["toff"]},
        'AP_amplitude': {"stim_start": timings["HyperDepol"]["tmid"],
                         "stim_end": timings["HyperDepol"]["toff"]},
        'ISI_values': {"stim_start": timings["HyperDepol"]["tmid"],
                       "stim_end": timings["HyperDepol"]["toff"]},
        'sag_amplitude': {"stim_start": timings["HyperDepol"]["ton"],
                          "stim_end": timings["HyperDepol"]["tmid"]},
        'sag_ratio1': {"stim_start": timings["HyperDepol"]["ton"],
                       "stim_end": timings["HyperDepol"]["tmid"]},
        'sag_ratio2': {"stim_start": timings["HyperDepol"]["ton"],
                       "stim_end": timings["HyperDepol"]["tmid"]},
    }

    return hyper_depol_efeatures


def _get_sahp_efeatures(timings):
    sahp_efeatures = {
        'Spikecount': {"stim_start": timings["sAHP"]["tmid"],
                       "stim_end": timings["sAHP"]["tmid2"]},
        'AP_amplitude': {"stim_start": timings["sAHP"]["tmid"],
                         "stim_end": timings["sAHP"]["tmid2"]},
        'ISI_values': {"stim_start": timings["sAHP"]["tmid"],
                       "stim_end": timings["sAHP"]["tmid2"]},
        'AHP_depth': {"stim_start": timings["sAHP"]["tmid"],
                      "stim_end": timings["sAHP"]["tmid2"]},
        'AHP_depth_abs': {"stim_start": timings["sAHP"]["tmid"],
                          "stim_end": timings["sAHP"]["tmid2"]},
        'AHP_time_from_peak': {"stim_start": timings["sAHP"]["tmid"],
                               "stim_end": timings["sAHP"]["tmid2"]},
        'steady_state_voltage_stimend': {"stim_start": timings["sAHP"]["tmid"],
                                         "stim_end": timings["sAHP"]["tmid2"]},
    }
    return sahp_efeatures


def _get_poscheops_efeatures(timings):
    poscheops_efeatures = {
        'Spikecount': {"stim_start": timings["PosCheops"]["ton"],
                       "stim_end": timings["PosCheops"]["t1"]}
    }
    return poscheops_efeatures


def _get_poscheops_targets(timings):
    poscheops_targets = {
        "PosCheops1":
            {  # Used for validation, need to check exact timings
                "amplitudes": [300],
                "tolerances": [10],
                "efeatures": {'Spikecount': {"stim_start": timings["PosCheops"]["ton"],
                                             "stim_end": timings["PosCheops"]["t1"]}},
                "location": "soma"
            },
        "PosCheops2":
            {  # Used for validation, need to check exact timings
                "amplitudes": [300],
                "tolerances": [10],
                "efeatures": {'Spikecount': {"stim_start": timings["PosCheops"]["t2"],
                                             "stim_end": timings["PosCheops"]["t3"]}},
                "location": "soma"
            },
        "PosCheops3":
            {  # Used for validation, need to check exact timings
                "amplitudes": [300],
                "tolerances": [10],
                "efeatures": {'Spikecount': {"stim_start": timings["PosCheops"]["t4"],
                                             "stim_end": timings["PosCheops"]["toff"]}},
                "location": "soma"
            },
    }

    return poscheops_targets


def get_ecode_targets(timings):
    targets = {
        "IDthres": {
            "amplitudes": [],  # Not used to extract actual e-features, just to compute the rheobase
            "tolerances": [10],
            "efeatures": [],
            "location": "soma"
        },
        "firepattern": {
            "amplitudes": [120, 200],  # Amplitudes will have to change if the rheobase shifted.
            "tolerances": [20],
            "efeatures": [
                'mean_frequency',
                'burst_number',
                'adaptation_index2',
                'ISI_CV',
                'ISI_log_slope',
                'inv_time_to_first_spike',
                'inv_first_ISI',
                'inv_second_ISI',
                'inv_third_ISI',
                'inv_fourth_ISI',
                'inv_fifth_ISI',
                'AP_amplitude',
                'AHP_depth',
                'AHP_time_from_peak',
            ],
            "location": "soma"
        },
        "IDrest": {  # Not sure what to use it for, except to get the IF curve.
            "amplitudes": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
            "tolerances": [20],
            "efeatures": [
                'mean_frequency',
                'burst_number',
                'adaptation_index2',
                'ISI_CV',
                'ISI_log_slope',
                'inv_time_to_first_spike',
                'inv_first_ISI',
                'inv_second_ISI',
                'inv_third_ISI',
                'inv_fourth_ISI',
                'inv_fifth_ISI',
                'AP_amplitude',
                'AHP_depth',
                'AHP_time_from_peak',
            ],
            "location": "soma"
        },
        "IV": {
            "amplitudes": [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60],  # -100 for the sag, -20 for the "passives", 0 for the RMP
            "tolerances": [10],
            "efeatures": [
                'Spikecount',
                'voltage_base',
                'voltage_deflection',
                'voltage_deflection_begin',
                'steady_state_voltage_stimend',
                'ohmic_input_resistance_vb_ssse',
                'sag_amplitude',
                'sag_ratio1',
                'sag_ratio2',
                'decay_time_constant_after_stim',
            ],
            "location": "soma"
        },
        "APWaveform": {
            "amplitudes": [200, 230, 260, 290, 320, 350],  # Arbitrary choice
            "tolerances": [20],
            "efeatures": [
                'AP_amplitude',
                'AP1_amp',
                'AP2_amp',
                'AP_duration_half_width',
                'AP_begin_width',
                'AP_begin_voltage',
                'AHP_depth',
                'AHP_time_from_peak'
            ],
            "location": "soma"
        },
        "HyperDepol": {  # Used for validation
            "amplitudes": [-160, -120, -80, -40],  # Arbitrary choice
            "tolerances": [10],
            "efeatures": _get_hyper_depol_efeatures(timings),
            "location": "soma"
        },
        "sAHP": {
            # Used for validation, It's not obvious in Mikael's schema if the percentage is relative to the base or to the first step
            "amplitudes": [150, 200, 250, 300],  # Arbitrary choice
            "tolerances": [10],
            "efeatures": _get_sahp_efeatures(timings),
            "location": "soma"
        },
        "PosCheops":
            {  # Used for validation, need to check exact timings
                "amplitudes": [300],
                "tolerances": [10],
                "efeatures": _get_poscheops_efeatures(timings),
                "location": "soma"
            }
    }

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

    feature_set = "soma"
    out_efeatures[feature_set] = {}

    out_protocols = {}
    out_efeatures = {"soma": {}}

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
                out_efeatures[feature_set][protocol_name] = {loc_name: efeatures_def}

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
                                 single_channel_features=False, std_from_mean=None):
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
        If 'single_channel_features' is True,


    Returns
    -------
    efeatures_dict: dict
       Modified dictionary with efeatures dict, including MEA features, saved to json

    """
    feature_set = "extra"
    new_efeatures_dict = efeatures_dict

    available_feature_sets = list(efeatures_dict.keys())
    assert len(available_feature_sets) == 1
    new_efeatures_dict[feature_set] = deepcopy(efeatures_dict[available_feature_sets[0]])

    if protocol_name not in list(new_efeatures_dict[feature_set].keys()):
        # create protocol if non existing
        new_efeatures_dict[feature_set][protocol_name] = {}

    # create MEA location
    new_efeatures_dict[feature_set][protocol_name]["MEA"] = {}

    for extra_feat_name, feature_values in extra_features.items():
        feature_list = feature_values.tolist()
        if channel_ids is None:
            new_efeatures_dict[feature_set][protocol_name]["MEA"][extra_feat_name] = [feature_list, None]
        elif isinstance(channel_ids, list) and np.isscalar(channel_ids[0]) and single_channel_features:
            # save channels separately
            assert std_from_mean is not None, "When 'single_channel_features' is used, the 'std_from_mean' argument " \
                                              "should be specified"
            for chan in channel_ids:
                new_efeatures_dict[feature_set][protocol_name]["MEA"][f"{extra_feat_name}_{chan}"] = \
                    [feature_list[chan], np.abs(std_from_mean * feature_list[chan])]
        else:
            if np.isscalar(channel_ids[0]):
                # subset
                new_efeatures_dict[feature_set][protocol_name]["MEA"][extra_feat_name] = \
                    [list(np.array(feature_list)[channel_ids]), channel_ids]
            else:
                for sec, channels in enumerate(channel_ids):
                    new_efeatures_dict[feature_set][protocol_name]["MEA"][f"{extra_feat_name}_{sec}"] = \
                        [list(np.array(feature_list)[channels]), channels]

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