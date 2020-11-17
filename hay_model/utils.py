import numpy as np


## HELPER FUNCTIONS ##
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

    b, a = ss.butter(order, band, btype=btype)

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
