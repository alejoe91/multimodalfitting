import numpy as np
import json
import os

import bluepyopt.ephys as ephys
import time


# Helper function to turn feature dicitonary into a list
def vectorize_features(feature_list):
    """
    Vectorizes feature dictionary into a list

    Parameters
    ----------
    feature_list: list
        List with features dictionaries from different responses organized as:
            - protocol
                - location
                    - feature_name: [mean, std]

    Returns
    -------
    feature_vectors: list
        List with vectorized features dictionaries for each response organized as:
            - protocol.location.feature_name: mean
    """
    feature_vectors = []
    for feature in feature_list:
        feature_vector = {}
        for prot, prot_dict in feature.items():
            for loc, loc_feat in prot_dict.items():
                for feat, feat_val in loc_feat.items():
                    feature_vector[f'{prot}.{loc}.{feat}'] = feat_val[0]
        feature_vectors.append(feature_vector)
    return feature_vectors


def compute_feature_values(params, cell_model, protocols, sim, feature_set='bap', std=0.2,
                           probe=None, channels="map", detect_threshold=0, verbose=False):
    """
    Calculate features for cell model and protocols.

    Parameters
    ----------
    params: pd.Series
        The parameters to be used (the index of the series is the param name
    cell_model: LFPyCellModel
        The cell model
    protocols: list
        List of protocols to extract features from
    sim: Simulator
        The BPO simulator
    feature_set: str
        'soma', 'multiple', 'extra', 'all'
    std: float
        The percent of the mean to use as standard deviation (default 0.2)
    probe: MEAutility.MEA
        The probe to use for extracellular features
    channels: list, "map", or None
        If None, features are computed separately for each channel
        If list, features are computed separately for the provided channels
        If 'map' (default), each feature is an array with the features computed on all channels

    Returns
    -------
    responses: dict
        Dictionary with BPO responses to specified protocols (protocol names are keys)
    features: dict
        Dictionary with extracted features organized as:
            - protocol_name
                - location
                    - feature_name: [mean, std]

    """
    assert feature_set in ['multiple', 'soma', 'extra', 'all']
    from configs import config_dir

    feature_list = json.load(
        open(os.path.join(config_dir, 'features_list.json')))[feature_set]

    if feature_set in ['extra', 'all']:
        assert probe is not None, "Provide a MEAutility probe to use the 'extra' set"

    if channels is None:
        channels = np.arange(probe.number_electrodes)

    features = {}

    for protocol_name, locations in feature_list.items():
        features[protocol_name] = []
        for location, feats in locations.items():
            for efel_feature_name in feats:

                if protocol_name in protocols:

                    feature_name = '%s.%s.%s' % (
                        protocol_name, location, efel_feature_name)
                    kwargs = {}

                    stimulus = protocols[protocol_name].stimuli[0]
                    kwargs['stim_start'] = stimulus.step_delay

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
                        kwargs['extrafel_feature_name'] = efel_feature_name
                        if channels != 'map':
                            for ch in channels:
                                kwargs['channel_id'] = int(ch)
                                feature = feature_class(
                                    feature_name,
                                    exp_mean=0,
                                    exp_std=0,
                                    **kwargs)
                                features[protocol_name].append(feature)
                        else:
                            kwargs['channel_id'] = None
                            feature = feature_class(
                                feature_name,
                                exp_mean=None,
                                exp_std=None,
                                **kwargs)
                            features[protocol_name].append(feature)
                    else:
                        feature_class = ephys.efeatures.eFELFeature
                        kwargs['efel_feature_name'] = efel_feature_name
                        kwargs['recording_names'] = {'': '%s.%s.v' % (protocol_name, location)}

                        feature = feature_class(
                            feature_name,
                            exp_mean=0,
                            exp_std=0,
                            **kwargs)
                        features[protocol_name].append(feature)

    responses = {}

    for protocol_name, protocol in protocols.items():
        if verbose:
            print('Running', protocol_name)
        t1 = time.time()
        responses.update(protocol.run(cell_model=cell_model, param_values=params, sim=sim))
        if verbose:
            print(time.time() - t1)

    feature_meanstd = {}
    for protocol_name, featlist in features.items():
        if verbose:
            print(protocol_name, 'Num features:', len(featlist))

        mean_std = {}
        for feat in featlist:
            prot, location, name = feat.name.split('.')

            if location != 'MEA':
                val = feat.calculate_feature(responses)
                if val is not None:
                    if location not in mean_std.keys():
                        mean_std[location] = {}
                    mean_std[location][name] = [val, np.abs(std * val)]
                else:
                    if verbose:
                        print(f"Feature {name} at {location} is None")
            else:
                if channels != 'map':
                    val = feat.calculate_feature(responses)
                    if val is not None and val != 0:
                        if isinstance(feat, ephys.efeatures.eFELFeature):
                            feat_name = name
                        else:
                            feat_name = f'{name}_{str(feat.channel_id)}'
                        if location not in mean_std.keys():
                            mean_std[location] = {}
                        mean_std[location][feat_name] = [val, np.abs(std * val)]
                else:
                    val = feat.calculate_feature(responses, detect_threshold=detect_threshold)
                    if location not in mean_std.keys():
                        mean_std[location] = {}
                    mean_std[location][name] = [val, None]

        feature_meanstd[protocol_name] = mean_std

    return responses, feature_meanstd


def calculate_eap(responses, protocol_name, protocols, sweep_id=0, fs=20, fcut=1,
                  ms_cut=[2, 10], filt_type="filtfilt", skip_first_spike=True, skip_last_spike=True,
                  raise_warnings=False, verbose=False, **efel_kwargs):
    """
    Calculate extracellular action potential (EAP) by combining intracellular spike times and extracellular signals

    Parameters
    ----------
    responses: dict
        Dictionary with BPO responses
    protocol_name: str
        The protocol to use to compute the EAP (must be a Step)
    protocols: list
        The list of protocols (used to extract stimuli information)
    sweep_id: int
        In case of sweep protocols, the sweep id to use.
    fs: float
        The final sampling frequency in kHz (default 20 kHz)
    fcut: float or list of 2 floats
        If float, the high-pass filter cutoff
        If list of 2 floats, the high-pass and low-pass filter cutoff
    ms_cut: list of 2 floats
        Ms to cutout before and after peak
    filt_type: str
        Filter type:
            - 'lfilter': forward pass
            - 'filtfilt': forward-backward pass
    skip_first_spike: bool
        If True, the first spike is not used to compute the EAP
    skip_last_spike: bool
        If True, the last spike is not used to compute the EAP
    raise_warnings: bool
        If True, eFEL raise warnings
    verbose: bool
        If True, output is verbose
    efel_kwargs: kwargs
        Keyword arguments for eFEL


    Returns
    -------
    eap: np.array
        The EAP (num_elec, num_samples)
    """
    protocol_responses = [resp for resp in responses.keys() if protocol_name in resp]

    if len(protocol_responses) > 1:
        protocol = protocols[protocol_name].protocols[sweep_id]
        response_name = f"{protocol_name}-{sweep_id}"
    else:
        protocol = protocols[protocol_name]
        response_name = protocol_name

    stimulus = protocol.stimuli[0]
    stim_start = stimulus.step_delay
    stim_end = stimulus.step_delay + stimulus.step_duration
    efel_kwargs['threshold'] = -20

    somatic_recording_name = f'{response_name}.soma.v'
    extra_recording_name = f'{response_name}.MEA.LFP'

    assert somatic_recording_name in responses.keys(), f"{somatic_recording_name} not found in responses"
    assert extra_recording_name in responses.keys(), f"{extra_recording_name} not found in responses"

    peak_times = _get_peak_times(responses, somatic_recording_name, stim_start, stim_end,
                                 raise_warnings=raise_warnings, **efel_kwargs)

    if len(peak_times) > 1 and skip_first_spike:
        peak_times = peak_times[1:]

    if len(peak_times) > 1 and skip_last_spike:
        peak_times = peak_times[:-1]

    if responses[extra_recording_name] is not None:
        response = responses[extra_recording_name]
    else:
        return None

    if np.std(np.diff(response['time'])) > 0.001 * np.mean(np.diff(response['time'])):
        assert fs is not None
        if verbose:
            print('interpolate')
        response_interp = _interpolate_response(response, fs=fs)
    else:
        response_interp = response

    if fcut is not None:
        if verbose:
            print('filter enabled')
        response_filter = _filter_response(response_interp, fcut=fcut, filt_type=filt_type)
    else:
        if verbose:
            print('filter disabled')
        response_filter = response_interp

    ewf = _get_waveforms(response_filter, peak_times, ms_cut)
    mean_wf = np.mean(ewf, axis=0)

    return mean_wf


## HELPER FUNCTIONS FOR EAP##
def _construct_somatic_efel_trace(
    responses, somatic_recording_name, stim_start, stim_end
):
    """Construct trace that can be passed to eFEL"""

    trace = {}
    if somatic_recording_name not in responses:
        return None

    if responses[somatic_recording_name] is None:
        return None

    response = responses[somatic_recording_name]

    trace["T"] = response["time"]
    trace["V"] = response["voltage"]
    trace["stim_start"] = [stim_start]
    trace["stim_end"] = [stim_end]

    return trace


def _setup_efel(
    threshold=None, interp_step=None, double_settings=None, int_settings=None
):
    """Set up efel before extracting the feature"""

    import efel

    efel.reset()

    if threshold is not None:
        efel.setThreshold(threshold)

    if interp_step is not None:
        efel.setDoubleSetting("interp_step", interp_step)

    if double_settings is not None:
        for setting_name, setting_value in double_settings.items():
            efel.setDoubleSetting(setting_name, setting_value)

    if int_settings is not None:
        for setting_name, setting_value in int_settings.items():
            efel.setIntSetting(setting_name, setting_value)


def _get_peak_times(
    responses,
    somatic_recording_name,
    stim_start,
    stim_end,
    raise_warnings=False,
    **efel_kwargs
):
    efel_trace = _construct_somatic_efel_trace(
        responses, somatic_recording_name, stim_start, stim_end
    )

    if efel_trace is None:
        peak_times = None
    else:
        _setup_efel(**efel_kwargs)

        import efel

        peaks = efel.getFeatureValues(
            [efel_trace], ["peak_time"], raise_warnings=raise_warnings
        )
        peak_times = peaks[0]["peak_time"]

        efel.reset()

    return peak_times


def _interpolate_response(response, fs=20.0):
    from scipy.interpolate import interp1d

    x = response["time"]
    y = response["voltage"]
    f = interp1d(x, y, axis=1)
    xnew = np.arange(np.min(x), np.max(x), 1.0 / fs)
    ynew = f(xnew)  # use interpolation function returned by `interp1d`

    response_new = {}
    response_new["time"] = xnew
    response_new["voltage"] = ynew

    return response_new


def _filter_response(response, fcut=[0.5, 6000], order=2, filt_type="lfilter"):
    import scipy.signal as ss

    fs = 1 / np.mean(np.diff(response["time"])) * 1000
    fn = fs / 2.0

    trace = response["voltage"]

    if isinstance(fcut, (float, int, np.float, np.integer)):
        btype = "highpass"
        band = fcut / fn
    else:
        assert isinstance(fcut, (list, np.ndarray)) and len(fcut) == 2
        btype = "bandpass"
        band = np.array(fcut) / fn

    b, a = ss.butter(order, fcut, btype=btype, fs=fs)

    if len(trace.shape) == 2:
        if filt_type == "filtfilt":
            filtered = ss.filtfilt(b, a, trace, axis=1)
        else:
            filtered = ss.lfilter(b, a, trace, axis=1)
    else:
        if filt_type == "filtfilt":
            filtered = ss.filtfilt(b, a, trace)
        else:
            filtered = ss.lfilter(b, a, trace)

    response_new = {}
    response_new["time"] = response["time"]
    response_new["voltage"] = filtered

    return response_new


def _upsample_wf(waveforms, upsample):
    from scipy.signal import resample_poly

    ndim = len(waveforms.shape)
    waveforms_up = resample_poly(waveforms, up=upsample, down=1, axis=ndim - 1)

    return waveforms_up


def _get_waveforms(response, peak_times, snippet_len_ms):
    times = response["time"]
    traces = response["voltage"]

    assert np.std(np.diff(times)) < 0.001 * np.mean(
        np.diff(times)
    ), "Sampling frequency must be constant"

    fs = 1.0 / np.mean(np.diff(times))  # kHz

    reference_frames = (peak_times * fs).astype(int)

    if isinstance(snippet_len_ms, (tuple, list, np.ndarray)):
        snippet_len_before = int(snippet_len_ms[0] * fs)
        snippet_len_after = int(snippet_len_ms[1] * fs)
    else:
        snippet_len_before = int((snippet_len_ms + 1) / 2 * fs)
        snippet_len_after = int((snippet_len_ms - snippet_len_before) * fs)

    num_snippets = len(peak_times)
    if len(traces.shape) == 2:
        num_channels = traces.shape[0]
    else:
        num_channels = 1
        traces = traces[np.newaxis, :]
    num_frames = len(times)
    snippet_len_total = int(snippet_len_before + snippet_len_after)
    waveforms = np.zeros(
        (num_snippets, num_channels, snippet_len_total), dtype=traces.dtype
    )

    for i in range(num_snippets):
        snippet_chunk = np.zeros((num_channels, snippet_len_total), dtype=traces.dtype)
        if 0 <= reference_frames[i] < num_frames:
            snippet_range = np.array(
                [
                    int(reference_frames[i]) - snippet_len_before,
                    int(reference_frames[i]) + snippet_len_after,
                ]
            )
            snippet_buffer = np.array([0, snippet_len_total], dtype="int")
            # The following handles the out-of-bounds cases
            if snippet_range[0] < 0:
                snippet_buffer[0] -= snippet_range[0]
                snippet_range[0] -= snippet_range[0]
            if snippet_range[1] >= num_frames:
                snippet_buffer[1] -= snippet_range[1] - num_frames
                snippet_range[1] -= snippet_range[1] - num_frames
            snippet_chunk[:, snippet_buffer[0] : snippet_buffer[1]] = traces[
                :, snippet_range[0] : snippet_range[1]
            ]
        waveforms[i] = snippet_chunk

    return waveforms