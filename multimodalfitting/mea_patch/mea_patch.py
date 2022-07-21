import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st

import numpy as np
import neo
import efel
from pathlib import Path
import time
import shutil
import matplotlib.pyplot as plt


def get_peak_times(patch, data_channel=0, efel_threshold=-20, mode="peak"):
    """
    Get peak times form patch data

    Parameters
    ----------
    patch: patch dict
        Patch data dictionary obtained when loading the data
    data_channel: int
        The channel containing the voltage trace
    efel_threshold: float
        EFEL threshold to find the peak (default -20)
    mode: str
        "peak", "begin", "rise" (default "peak")

    Returns
    -------
    peak_times: np.array
        Peak times in s

    """
    assert mode in ["peak", "begin", "rise"]

    # construct efel trace
    trace = {}
    trace['T'] = patch['time'] * 1000  # to ms
    trace['V'] = patch['data'][data_channel]
    trace['stim_start'] = [patch['time'][0]]
    trace['stim_end'] = [patch['time'][-1]]

    efel.setThreshold(efel_threshold)
    if mode == "peak":
        peaks = efel.getFeatureValues([trace], ['peak_time'])
        peak_times = peaks[0]['peak_time'] / 1000  # to s
    elif mode == "begin":
        peaks = efel.getFeatureValues([trace], ['AP_begin_time'])
        peak_times = peaks[0]['AP_begin_time'] / 1000  # to s
    else:
        peaks = efel.getFeatureValues([trace], ['AP_rise_time'])
        peak_times = peaks[0]['AP_rise_time'] / 1000  # to s

    return peak_times


def remove_blank_channels(recording, n_segments=10, seconds_per_segments=1):
    """
    Removes blank channels from recording.
    'n_segments' of traces with 'seconds_per_segment' duration are extracted from the recording.
    Channels that are blank (diff = 0) in any of the segment are removed.

    Parameters
    ----------
    recording: RecordingExtractor
        The input recording extractor
    n_segments: int
        Number of segments to extract (default 10)
    seconds_per_segments: float
        Seconds per segment (default 1)

    Returns
    -------
    rec_clean: RemoveBadChannelsRecording
        The output recording without blank channels
    """
    fs = recording.get_sampling_frequency()
    samples_per_segment = int(seconds_per_segments * fs)
    segment_start = np.linspace(0, recording.get_num_frames() - samples_per_segment - 1, n_segments).astype(int)

    remove_set = set([])
    for seg_start in segment_start:
        traces = recording.get_traces(start_frame=seg_start, end_frame=seg_start + samples_per_segment)
        blank_channels_in_segment = []
        for ch_idx, tr in enumerate(traces.T):
            if np.all(np.diff(tr) == 0):
                chan_to_remove = recording.get_channel_ids()[ch_idx]
                blank_channels_in_segment.append(chan_to_remove)
        remove_set = remove_set | set(blank_channels_in_segment)
    remove_channels = list(remove_set)
    print(f"Removing {len(remove_channels)} blank channels")
    keep_channels = list(set(recording.get_channel_ids()) - remove_set)
    rec_clean = recording.channel_slice(channel_ids=keep_channels)

    return rec_clean


def get_corrected_timestamps(recording, verbose=False):
    """
    Corrects for missing frames.
    functions.

    Parameters
    ----------
    recording: MaxwellRecording
    verbose: bool
        If True, output is verbose

    Returns
    -------
    timestamps: np.array
        Timestamps in seconds
    """
    def get_frame_number(recording, index, version):
        stream_id = recording.stream_id
        if int(version) == 20160704:
            bitvals = recording.neo_reader._signals[stream_id][-2:, index]
            frameno = bitvals[1] << 16 | bitvals[0]
        elif int(version) > 20160704:
            frameno = \
                recording.neo_reader.h5_file['wells'][stream_id][recording.neo_reader.rec_name]['groups']['routed'][
                    "frame_nos"][index]
        return frameno

    def get_all_frame_numbers(recording, version):
        stream_id = recording.stream_id
        if int(version) == 20160704:
            bitvals = recording.neo_reader._signals[stream_id][-2:, :]
            frame_nos = np.bitwise_or(np.left_shift(bitvals[-1].astype('int64'), 16), bitvals[0])
        elif int(version) > 20160704:
            frame_nos = \
            recording.neo_reader.h5_file['wells'][stream_id][recording.neo_reader.rec_name]['groups']['routed'][
                "frame_nos"][:]
        return frame_nos

    assert isinstance(recording, se.MaxwellRecordingExtractor)
    timestamps = np.round(np.arange(recording.get_num_frames()) / recording.get_sampling_frequency(), 6)
    version = recording.neo_reader.h5_file['version'][0].decode()

    frame_idxs_span = get_frame_number(recording, recording.get_num_frames() - 1, version) - \
                      get_frame_number(recording, 0, version)

    if frame_idxs_span > recording.get_num_frames():
        if verbose:
            print(f"Found missing frames! Correcting for it (this might take a while)")

        framenos = get_all_frame_numbers(recording, version)
        # find missing frames
        diff_frames = np.diff(framenos)
        missing_frames_idxs = np.where(diff_frames > 1)[0]

        delays_in_frames = []
        for mf_idx in missing_frames_idxs:
            delays_in_frames.append(diff_frames[mf_idx])

        if verbose:
            print(f"Found {len(delays_in_frames)} missing intervals.\nLength missing frames: {delays_in_frames}")

        timestamps = np.round(np.arange(recording.get_num_frames()) / recording.get_sampling_frequency(), 6)

        for mf_idx, duration in zip(missing_frames_idxs, delays_in_frames):
            timestamps[mf_idx:] += np.round(duration / recording.get_sampling_frequency(), 6)

    else:
        if verbose:
            print("No missing frames found")

    return timestamps


def sync_patch_mea(recording_file, patch_files, patch_ttl_channel=None, patch_ttl_delay=None,
                   correct_mea_times=True, n_seconds=None, verbose=True,
                   remove_blank_mea_channels=True, 
                   return_patch_single_sweeps=False, 
                   return_mea_single_sweeps=False):
    '''
    Synchronize MEA and patch file(s) using TTL pulses

    Parameters
    ----------
    recording_file: Path or str
        Path to Mea1k/Maxwell file
    patch_files: list or str
        Path to patch .wcp file(s) associated to the MEA recording
    patch_ttl_channel: int
        Wcp channel with TTL pulses
    patch_ttl_delay: float
        Constant delay in seconds for each sweep to first TTL (if patch_ttl_channel is None)
    correct_mea_times: bool
        If True (default), mea timestamps are corrected for missing frames
    remove_blank_mea_channels: bool
        If True (default), blank/saturating channels are removed from MEA recording before processing
    n_seconds: float
        Cuts out n_seconds of the MEA recording (for testing purposes)
    return_patch_single_sweeps: bool

    verbose: bool
        If True, output is verbose

    Returns
    -------
    rec: SubRecordingExtractor
        Synced recording extractor
    patch: dict
        Dictionary with patch data:
            - time: np.array with synced times in s
            - data: np.array (num_patch_channels x time_samples)
    sync_bits: np.array
        Array with synced TTL mea times
    '''
    assert patch_ttl_channel is not None or patch_ttl_delay is not None

    if isinstance(patch_files, (str, Path)):
        patch_files = [str(patch_files)]

    # load recording
    recording = se.MaxwellRecordingExtractor(recording_file)
    event = se.MaxwellEventExtractor(recording_file)

    # load bit times
    if correct_mea_times:
        print("Correcting for missing frames")
        timestamps = get_corrected_timestamps(recording, verbose=True)
    else:
        timestamps = np.arange(recording.get_num_samples()) / recording.get_sampling_frequency()

    events = event.get_event_times()
    ttl_rising = events[events["state"] > 0]
    if len(np.unique(ttl_rising["state"])) > 1:
        print(f"More than a MEA TTL channel is found. Use the 'mea_ttl_channel' argument to specify "
              f"which one to use: {np.unique(ttl_rising['state'])}")
    start_frame = ttl_rising[0]["frame"]

    # TTLs don't suffer from missing frames
    time_tp_mea = ttl_rising["time"]  #/ recording.get_sampling_frequency()

    # get sweep times
    tp_delta_mea = np.diff(time_tp_mea) * 1000
    idx_sweep_tp2_mea = np.where(tp_delta_mea < tp_delta_mea[0] + 5)[0]
    idx_sweep_tp_mea = idx_sweep_tp2_mea[[True] + list(np.diff(idx_sweep_tp2_mea) > 1)]
    time_sweep_tp_mea = time_tp_mea[idx_sweep_tp_mea]

    if remove_blank_mea_channels:
        recording_clean = remove_blank_channels(recording)
    else:
        recording_clean = recording

    if verbose:
        print(f'Found sweep times in MEA: {len(list(time_sweep_tp_mea))}')

    tp_patch_global = []
    patch_data_global = []
    patch_files_global = []
    cont_mea_ttl = 0

    # load patch data
    for patch_file in patch_files:
        io = neo.WinWcpIO(str(patch_file))
        block = io.read_block()
        segments = block.segments
        tot_patch_sweeps = len(segments)
        num_patch_channels = 0
        for anas in segments[0].analogsignals:
            num_patch_channels += anas.shape[1]
        seg_len = len(segments[0].analogsignals[0])
        num_anas = len(segments[0].analogsignals)
        num_channels_per_anas = [anas.shape[1] for anas in segments[0].analogsignals]

        if verbose:
            print(f'Found {tot_patch_sweeps} sweeps in PATCH')

        tot_sweeps = tot_patch_sweeps

        if not return_patch_single_sweeps:
            # concatenate patch times and data
            tp_patch = np.zeros(tot_sweeps * seg_len)
            patch_data = np.zeros((num_patch_channels, tot_sweeps * seg_len))

            for i in range(tot_sweeps):
                seg = segments[i]
                mea_ttl_idx = i + cont_mea_ttl

                if i == 0:
                    sampling_rate = seg.analogsignals[0].sampling_rate

                seg_data = np.zeros((num_patch_channels, seg_len))
                tp_patch_seg = seg.analogsignals[0].times.rescale('s').magnitude
                ch = 0
                for i_analog in range(num_anas):
                    analog = seg.analogsignals[i_analog].as_array()
                    for i_ch in range(num_channels_per_anas[i_analog]):
                        seg_data[ch] = analog[:, i_ch]
                        patch_data[ch, i * seg_len:i * seg_len + seg_len] = seg_data[ch]
                        ch += 1

                if patch_ttl_channel is not None:
                    seg_ttl = seg_data[patch_ttl_channel]

                    # find all test pulses
                    thresh = 1000  # Test pulse threshold; TP peak is ~3000 mV
                    idxl = np.where(seg_ttl >= thresh)[0]
                    yest = np.where(seg_ttl[idxl - 1] < thresh)[0]
                    time_tp_patch = tp_patch_seg[idxl[yest]]
                    tp_patch[i * seg_len:i * seg_len + seg_len] = tp_patch_seg + time_sweep_tp_mea[mea_ttl_idx] \
                                                                  - time_tp_patch[0]

                else:
                    tp_patch_seg = seg.analogsignals[0].times.rescale('s').magnitude
                    tp_patch[i * seg_len:i * seg_len + seg_len] = tp_patch_seg + time_sweep_tp_mea[mea_ttl_idx] \
                                                                  - patch_ttl_delay
        else:
            # return a list of patch times and data
            tp_patch = []
            patch_data = []

            for i in range(tot_sweeps):
                seg = segments[i]
                seg_data = np.zeros((num_patch_channels, seg_len))
                tp_patch_seg = seg.analogsignals[0].times.rescale('s').magnitude
                ch = 0
                for i_analog in range(num_anas):
                    analog = seg.analogsignals[i_analog].as_array()
                    for i_ch in range(num_channels_per_anas[i_analog]):
                        seg_data[ch] = analog[:, i_ch]
                        ch += 1

                if patch_ttl_channel is not None:
                    seg_ttl = seg_data[patch_ttl_channel]

                    # find all test pulses
                    thresh = 1000  # Test pulse threshold; TP peak is ~3000 mV
                    idxl = np.where(seg_ttl >= thresh)[0]
                    yest = np.where(seg_ttl[idxl - 1] < thresh)[0]
                    time_tp_patch = tp_patch_seg[idxl[yest]]
                    tp_patch_sync = tp_patch_seg + time_sweep_tp_mea[cont_mea_ttl + i] - time_tp_patch[0]
                    
                    # print(f"{patch_file.name} - sweep {i} - tp_patch_sync: {tp_patch_sync[0]} - TTL pulse: {time_sweep_tp_mea[cont_mea_ttl + i]}")
                    sampling_rate = seg.analogsignals[0].sampling_rate
                else:
                    tp_patch_sync = tp_patch_seg + time_sweep_tp_mea[cont_mea_ttl + i] - patch_ttl_delay
                    sampling_rate = seg.analogsignals[0].sampling_rate

                patch_data.append(seg_data)
                tp_patch.append(tp_patch_sync)

        cont_mea_ttl += tot_sweeps
        tp_patch_global.append(tp_patch)
        patch_data_global.append(patch_data)
        patch_files_global.append([patch_file.name] * len(tp_patch))
        
    if not return_patch_single_sweeps:
        tp_patch = np.array([])
        patch_data = np.empty((0, len(patch_data_global[0])))

        for tp, pdata in zip(tp_patch_global, patch_data_global):
            tp_patch = np.concatenate((tp_patch, tp))
            patch_data = np.vstack((patch_data, pdata.T))
        patch_data = patch_data.T

        # start time at 0 for both MEA and patch
        t_idxs_patch = tp_patch > time_sweep_tp_mea[0]
        tp_patch = tp_patch[t_idxs_patch]
        tp_patch -= tp_patch[0]
        patch_data = patch_data[:, t_idxs_patch]
    else:
        tp_patch = []
        patch_data = []
        patch_names = []

        ttl_index = 0
        
        for i, (tp_g, pd_g, pf_g) in enumerate(zip(tp_patch_global, patch_data_global, patch_files_global)):
            for j, (tp, pd, pf) in enumerate(zip(tp_g, pd_g, pf_g)):
                t_idxs_patch = tp > time_sweep_tp_mea[ttl_index]
                tp = tp[t_idxs_patch]
                tp -= tp[0]
                tp += time_sweep_tp_mea[ttl_index] - time_sweep_tp_mea[0]
                pd = pd[:, t_idxs_patch]
                ttl_index += 1
                tp_patch.append(tp)
                patch_data.append(pd)
                patch_names.append(pf)

    if not return_patch_single_sweeps:
        subrec = recording_clean.frame_slice(start_frame=start_frame, end_frame=recording_clean.get_num_samples())
        timestamps = timestamps[start_frame:]
        timestamps = timestamps - timestamps[0]

        mea_duration = timestamps[-1]

        if n_seconds is not None:
            if mea_duration > n_seconds:
                print(f"Clipping mea recording to {n_seconds}")
                end_frame = np.searchsorted(timestamps, n_seconds)
                subrec = subrec.frame_slice(start_frame=0, end_frame=end_frame)
                mea_duration = n_seconds
                timestamps = timestamps[:end_frame]

        patch_duration = tp_patch[-1]

        # clip at same time
        if mea_duration > patch_duration:
            print("Clipping MEA recording to match patch")
            end_frame = np.searchsorted(timestamps, patch_duration)
            subrec = subrec.frame_slice(start_frame=0, end_frame=end_frame)
            timestamps = timestamps[:end_frame]
            mea_duration = timestamps[-1]
        else:
            print("Clipping patch recording to match MEA")
            time_idxs = tp_patch <= mea_duration
            tp_patch = tp_patch[time_idxs]
            patch_data = patch_data[:, time_idxs]
            patch_duration = tp_patch[-1]
        
        subrec.set_times(timestamps)

        ttl_mea_sync = time_tp_mea - time_tp_mea[0]
    else:
        if return_mea_single_sweeps:
            subrec = []
            subtimestamps = []
            mea_duration = []
            patch_duration = []
            for t_i, ttl_mea in enumerate(time_sweep_tp_mea):
                start_frame_sweep = np.searchsorted(timestamps, ttl_mea)
                end_frame_sweep = start_frame_sweep + int(tp_patch[t_i][-1] * recording_clean.get_sampling_frequency())
                subrec_sweep = recording_clean.frame_slice(start_frame=start_frame_sweep, end_frame=end_frame_sweep)
                timestamps_sweep = timestamps[start_frame_sweep:end_frame_sweep]
                timestamps_sweep -= timestamps_sweep[0]

                if correct_mea_times:
                    mea_duration_sweep = timestamps_sweep[subrec_sweep.get_num_frames() - 1]
                else:
                    mea_duration_sweep = (subrec_sweep.get_num_frames() - 1) / subrec_sweep.get_sampling_frequency()

                tp_patch_sweep = tp_patch[t_i]
                patch_duration_sweep = tp_patch_sweep[-1]

                # clip at same time
                if mea_duration_sweep > patch_duration_sweep:
                    end_frame = np.searchsorted(timestamps_sweep, patch_duration_sweep)
                    subrec_sweep = subrec_sweep.frame_slice(start_frame=0, end_frame=end_frame)
                    timestamps = timestamps_sweep[:end_frame]
                    mea_duration_sweep = timestamps_sweep[-1]
                else:
                    time_idxs = tp_patch_sweep <= mea_duration_sweep
                    tp_patch_sweep = tp_patch_sweep[time_idxs]
                    patch_data_sweep = patch_data[t_i][:, time_idxs]
                    patch_duration_sweep = tp_patch_sweep[-1]
                    tp_patch[t_i] = tp_patch_sweep
                    patch_data[t_i] = patch_data_sweep

                subrec.append(subrec_sweep)
                subtimestamps.append(timestamps_sweep)
                mea_duration.append(mea_duration_sweep)
                patch_duration.append(patch_duration_sweep)

            timestamps = subtimestamps
        else:
            subrec = recording_clean.frame_slice(start_frame=start_frame, 
                                                 end_frame=recording_clean.get_num_samples())
            timestamps = timestamps[start_frame:]
            timestamps = timestamps - timestamps[0]
            
            if correct_mea_times:
                mea_duration = timestamps[subrec.get_num_frames() - 1]
            else:
                mea_duration = (subrec.get_num_frames() - 1) / subrec.get_sampling_frequency()
                
            patch_duration = tp_patch[-1][-1]

            # clip at same time
            if mea_duration > patch_duration:
                end_frame = np.searchsorted(timestamps, patch_duration)
                subrec = subrec.frame_slice(start_frame=0, end_frame=end_frame)
                timestamps = timestamps[:end_frame]
                mea_duration = timestamps[-1]
            else:
                time_idxs_last = tp_patch[-1] <= mea_duration
                tp_patch_last = tp_patch[-1][time_idxs_last]
                patch_data_last = patch_data[-1][:, time_idxs]
                patch_duration = tp_patch_last[-1]
                tp_patch[-1] = tp_patch_last
                patch_data[-1] = patch_data_last
            subrec.set_times(timestamps)
            
        ttl_mea_sync = time_tp_mea - time_tp_mea[0]

        if not return_patch_single_sweeps:
            ttl_mea_sync = ttl_mea_sync[ttl_mea_sync < mea_duration]

    print(f"MEA duration: {mea_duration}, Patch duration: {patch_duration}")
    if not return_patch_single_sweeps:
        patch = {'time': tp_patch, 'data': patch_data, "fs": sampling_rate}
    else:
        patch = []
        for (tp, pd, pn) in zip(tp_patch, patch_data, patch_names):
            patch_dict = {'time': tp, 'data': pd, "fs": sampling_rate, "name": pn}
            patch.append(patch_dict)

    return subrec, patch, timestamps, ttl_mea_sync


def load_patch_data(patch_files, average_sweeps=False, verbose=True):
    """
    Load the patch file(s) in a patch dictionary

    Parameters
    ----------
    patch_files: list of file paths or single file
        The patch file(s) to be loaded. If multiple files are given they are concatenated
    average_sweeps: bool
        If True, sweeps are averaged out
    verbose: bool
        If True, output is verbose

    Returns
    -------
    patch: dict
        Dictionary with patch data:
            - time: np.array with synced times in s
            - data: np.array (num_patch_channels x time_samples)
    """
    if isinstance(patch_files, (str, Path)):
        patch_files = [str(patch_files)]

    tp_patch_global = []
    patch_data_global = []

    for patch_file in patch_files:
        io = neo.WinWcpIO(str(patch_file))
        block = io.read_block()
        segments = block.segments
        tot_sweeps = len(segments)
        num_patch_channels = 0
        for anas in segments[0].analogsignals:
            num_patch_channels += anas.shape[1]
        seg_len = len(segments[0].analogsignals[0])
        num_anas = len(segments[0].analogsignals)
        num_channels_per_anas = [anas.shape[1] for anas in segments[0].analogsignals]

        if verbose:
            print(f'Found {tot_sweeps} sweeps in PATCH')

        if not average_sweeps:
            tp_patch = np.zeros(tot_sweeps * seg_len)
            patch_data = np.zeros((num_patch_channels, tot_sweeps * seg_len))

            last_patch_time = 0
            for i in range(tot_sweeps):
                seg = segments[i]
                seg_data = np.zeros((num_patch_channels, seg_len))
                tp_patch_seg = seg.analogsignals[0].times.rescale('s').magnitude
                if i == 0:
                    sampling_rate = seg.analogsignals[0].sampling_rate
                ch = 0
                for i_analog in range(num_anas):
                    analog = seg.analogsignals[i_analog].as_array()
                    for i_ch in range(num_channels_per_anas[i_analog]):
                        seg_data[ch] = analog[:, i_ch]
                        patch_data[ch, i * seg_len:i * seg_len + seg_len] = seg_data[ch]
                        ch += 1
                tp_patch[i * seg_len:i * seg_len + seg_len] = tp_patch_seg + last_patch_time
                last_patch_time += tp_patch_seg[-1]
        else:
            tp_patch = np.zeros(seg_len)
            patch_data = np.zeros((num_patch_channels, seg_len))

            for i in range(tot_sweeps):
                seg = segments[i]
                seg_data = np.zeros((num_patch_channels, seg_len))
                if i == 0:
                    tp_patch = seg.analogsignals[0].times.rescale('s').magnitude
                    sampling_rate = seg.analogsignals[0].sampling_rate
                ch = 0
                for i_analog in range(num_anas):
                    analog = seg.analogsignals[i_analog].as_array()
                    for i_ch in range(num_channels_per_anas[i_analog]):
                        seg_data[ch] = analog[:, i_ch]
                        patch_data[ch] += seg_data[ch]
                        ch += 1
                patch_data /= tot_sweeps

        tp_patch_global.append(tp_patch)
        patch_data_global.append(patch_data)

    tp_patch = np.array([])
    patch_data = np.empty((0, len(patch_data_global[0])))

    for tp, pdata in zip(tp_patch_global, patch_data_global):
        tp_patch = np.concatenate((tp_patch, tp))
        patch_data = np.vstack((patch_data, pdata.T))
    patch_data = patch_data.T

    patch = {'time': tp_patch, 'data': patch_data, 'fs': sampling_rate.magnitude}

    return patch


def patch_triggered_average(recording, patch, timestamps, save_folder, cut_out_ms=None, patch_data_channel=0,
                            center=True, electrode_id=None, radius=100, template_mode='median', peak_mode="peak",
                            center_ms=1.5, wf_indices=None, **wf_kwargs):
    """
    Compute patch triggered average from synchronized patch and MEA data.

    Parameters
    ----------
    recording: RecordingExtractor
        Synced extractor object from sync_patch_mea function
    patch: patch dict
        Synced patch dict from sync_patch_mea function
    timestamps: np.array
        Timestamps for recording extractor to properly align patch data
    save_folder: str or Path
        Folder used to save patch sorting and waveforms
    cut_out_ms: list or None
        Cut outs in ms for waveform extraction (default [2, 5] ms)
    patch_data_channel: int
        Pacth channel for voltage traces
    center: bool
        If True, waveforms are centered around the extracellular peak
    center_ms: float
        Interval to fine extracellular peak with respect to patch peak
    electrode_id: int or None
        If given, the peak is found in proximity of the electrode id
    radius: float
        If electrode_id is given
    template_mode: str
        Mode to extract template ("median" (default) - "mean")
    peak_mode: str
        Mode to compute peaks ("peak", "begin", "rise")
    wf_indices: array or None
        If given, only sample indices of the waveforms to use for constructing the template

    wf_kwargs: keyword arguments for si.extract_waveforms() function

    Returns
    -------
    sta: np.array
        Patch-triggered average template (num_samples, num_channels)
    waveforms: np.array
        Aligned waveforms (num_spikes, num_waveforms, num_channels)
    patch_sorting: SortingExtractor
        The sorting extractor with patch spike times
    waveform_extractor: WaveformExtractor
        The waveform extractor object
    patch_peaks: np.array
        Peak times of patch recording
    extra_peaks: np.array
        Peak times of extra recording (if center is True)
    """
    save_folder = Path(save_folder)
    if cut_out_ms is None:
        cut_out_ms = [2, 5]
    peak_times = get_peak_times(patch, data_channel=patch_data_channel, mode=peak_mode)

    print(f"Found {len(peak_times)} peaks")
    peak_frames = np.searchsorted(timestamps, peak_times).astype('int64')
    patch_sorting = si.NumpySorting.from_times_labels(times_list=peak_frames,
                                                      labels_list=np.array([0] * len(peak_frames)),
                                                      sampling_frequency=recording.get_sampling_frequency())

    patch_sorting_folder = save_folder / "patch_sorting"
    if patch_sorting_folder.is_dir():
        shutil.rmtree(patch_sorting_folder)
    patch_sorting = patch_sorting.save(folder=patch_sorting_folder)

    cut_out_ms = np.array(cut_out_ms)
    cut_out_ms_wide = cut_out_ms + 2 * center_ms
    cut_out_frames_wide = (cut_out_ms_wide * recording.get_sampling_frequency() / 1000).astype(int)
    cut_out_frames = (cut_out_ms * recording.get_sampling_frequency() / 1000).astype(int)

    print("Extracting uncentered waveforms")
    we_uncentered = si.extract_waveforms(recording, patch_sorting, folder=save_folder / "uncentered",
                                         ms_before=cut_out_ms_wide[0], ms_after=cut_out_ms_wide[1],
                                         overwrite=True, progress_bar=True, max_spikes_per_unit=None,
                                         **wf_kwargs)
    waveforms = we_uncentered.get_waveforms(unit_id=0)

    if wf_indices is not None:
        print(f"Using {len(wf_indices)} waveforms")
        waveforms_to_use = waveforms[wf_indices]
    else:
        waveforms_to_use = waveforms

    if center:
        print("Centering waveforms")
        if template_mode == 'median':
            sta = np.median(waveforms_to_use, axis=0)
        else:
            sta = np.mean(waveforms_to_use, axis=0)

        # find proximity electrodes
        if electrode_id is not None:
            all_channel_ids = recording.get_channel_ids()
            electrode_ids = recording.get_property("electrode")
            channel_id = all_channel_ids[list(electrode_ids).index(electrode_id)]
            loc_channel_id = recording.get_channel_locations(channel_ids=[channel_id])[0]
            distances = np.array([np.linalg.norm(loc - loc_channel_id) for loc in recording.get_channel_locations()])
            selected_channels = all_channel_ids[distances < radius]
            selected_channels_idxs = recording.ids_to_indices(selected_channels)
            sta_around_electrode = sta[:, selected_channels_idxs]
            selected_max_channel_idx = np.unravel_index(np.argmax(np.abs(sta_around_electrode)),
                                                        sta_around_electrode.shape)[1]
            max_channel = selected_channels[selected_max_channel_idx]
            max_channel_idx = recording.ids_to_indices([max_channel])[0]
        else:
            max_channel_idx = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)[1]
        print(f"Max channel idx: {max_channel_idx}")

        # center waveforms depending on idx on max channel
        waveforms_centered = np.zeros((waveforms_to_use.shape[0], np.sum(cut_out_frames), waveforms_to_use.shape[2]))
        samples_center = int(center_ms * recording.get_sampling_frequency() / 1000)
        center = cut_out_frames_wide[0]
        extrapeak_times = []
        for w_i, wf in enumerate(waveforms_to_use):
            min_idx = np.argmin(wf[center - samples_center: center + samples_center, max_channel_idx])
            extra_peak_frame = peak_frames[w_i] - samples_center + min_idx
            min_idx = cut_out_frames_wide[0] + min_idx - samples_center
            waveforms_centered[w_i] = wf[min_idx - cut_out_frames[0]: min_idx + cut_out_frames[1], :]
            extrapeak_times.append(timestamps[extra_peak_frame])
        waveforms_to_use = waveforms_centered
        extrapeak_times = np.array(extrapeak_times)
    else:
        cut_idx_start = cut_out_frames_wide[0] - cut_out_frames[0]
        cut_idx_end = cut_out_frames_wide[0] - cut_out_frames[0] + cut_out_frames[1]
        waveforms_to_use = waveforms_to_use[:, cut_idx_start:cut_idx_end]
        extrapeak_times = peak_times

    sta = np.median(waveforms_to_use, axis=0)

    return sta, waveforms_to_use, patch_sorting, we_uncentered, peak_times, extrapeak_times


def process_mea_patch(mea_file, patch_files, save_folder, save_recording=False,
                      patch_ttl_channel=None, patch_ttl_delay=None, freq_min=300, freq_max=6000,
                      notch_freq=None, center=True, center_electrode=None, center_ms=2, clean_sta=True,
                      plot_wf_hist=False, std_thresh=2, overwrite_recording=False,
                      overwrite_templates=False, filestem=None,
                      n_seconds=None, **wf_kwargs):
    """
    Process a MEA file and patch file(s) to compute patch-triggered average.

    First, MEA and patch data are synchronized, the MEA data is then filtered and optionally cached, and finally the
    template is computed. The template can also be cleaned to remove waveforms with amplitudes outside a certain std
    range.

    Parameters
    ----------
    mea_file: str or Path
        Path the mea h5 file
    patch_files: str, Path, or list
        Patch files or files. If a list, the files are concatenated
    save_folder: path
        The save folder
    save_recording: bool
        If True, the recording is saved in binary formato to the save_folder. Default False.
    patch_ttl_channel: int
        Wcp channel with TTL pulses
    patch_ttl_delay: float
        Constant delay in seconds for each sweep to first TTL (if patch_ttl_channel is None)
    freq_min: float
        Frequency for high pass filter (default 300)
    freq_max: float
        Frequency for low pass filter (default 6000)
    center: bool
        If True, waveforms are centered around the extracellular peak
    center_ms: float
        Interval to fine extracellular peak with respect to patch peak
    center_electrode: int or None
        If given, the peak is found in proximity of the electrode id
    clean_sta: bool
        If True, waveforms are cleaned by removing waveforms above and below std_thresh standard deviation from the
        median ammplitude
    std_thresh: float
        Std threshold to clean waveforms
    plot_wf_hist: bool
        If True, an histogram of the waveform amplitudes is plotted
    overwrite_recording: bool
        If True and recording folder already exist, it is overwritten
    overwrite_templates: bool
        If True and template files already exist, they are overwritten
    filestem: str
        Stem of the file to save templates and locations
    n_seconds: float
        Cuts out n_seconds of the MEA recording (for testing purposes)
    wf_kwargs: keyword arguments for si.extract_waveforms() function

    Returns
    -------
    template_save_path: Path
        Path to saved template npy file
    locations_save_path: Path
        Path to saved locations npy file

    """
    t_start = time.time()

    # save templates and locations
    save_folder = Path(save_folder)
    template_path = save_folder / "templates"
    if not template_path.is_dir():
        template_path.mkdir(parents=True, exist_ok=True)

    if isinstance(patch_files, (str, Path)):
        patch_files = [patch_files]

    if len(patch_files) == 1 and filestem is None:
        filestem = patch_files[0].stem
    else:
        assert filestem is not None, "When multiple patch files are used, 'filestem' must be specified"

    if n_seconds is None:
        template_save_path = template_path / f"{filestem}_template.npy"
        locations_save_path = template_path / f"{filestem}_locations.npy"
    else:
        template_save_path = template_path / f"{filestem}_{n_seconds}s_template.npy"
        locations_save_path = template_path / f"{filestem}_{n_seconds}s_locations.npy"

    if template_save_path.is_file() and locations_save_path.is_file() and not overwrite_templates:
        print("Returning already saved files!")
        return template_save_path, locations_save_path
    else:
        rec, patch, mea_ts, sync_bits = sync_patch_mea(mea_file, patch_files, patch_ttl_channel=patch_ttl_channel,
                                                       patch_ttl_delay=patch_ttl_delay, n_seconds=n_seconds,
                                                       verbose=True)

        mea_duration = mea_ts[-1]
        patch_duration = patch["time"][-1] - patch["time"][0]
        print(f"Duration MEA: {mea_duration} s - Duration patch: {patch_duration} s")

        if n_seconds is None:
            save_rec_folder = save_folder / f"{filestem}"
        else:
            print(f"Appending {n_seconds} to cache file")
            save_rec_folder = save_folder / f"{filestem}_{n_seconds}s"

        rec_f = st.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)

        # optional notch filter
        if notch_freq is not None:
            if np.isscalar(notch_freq):
                notch_freq = [notch_freq]
            else:
                assert isinstance(notch_freq, list)
            for nf in notch_freq:
                print(f"Applying notch filter at {nf} Hz")
                rec_f = st.preprocessing.notch_filter(rec_f, freq=nf)

        rec = rec_f

        if save_recording:
            if save_rec_folder.is_dir() and overwrite_recording:
                shutil.rmtree(save_rec_folder)
            if not save_rec_folder.is_dir():
                print(f"Saving recording to {save_rec_folder}")
                ts = time.time()
                rec = rec.save(folder=save_rec_folder, progress_bar=True, **wf_kwargs)
                te = time.time()
                print(f"Saving took {te - ts}")
            else:
                print(f"Loading recording from folder: {save_rec_folder}")
                rec = se.load_extractor(save_rec_folder)


        # compute aligned sta
        sta_raw, wf, patch_sorting, we, patch_peaks, extra_peaks = patch_triggered_average(rec, patch, mea_ts,
                                                                                           save_folder, center=center,
                                                                                           center_ms=center_ms,
                                                                                           electrode_id=center_electrode,
                                                                                           **wf_kwargs)

        if clean_sta:
            max_channel = np.unravel_index(np.argmax(np.abs(sta_raw)), sta_raw.shape)[1]
            wf_amp = np.max(np.abs(wf[:, :, max_channel]), axis=1)
            med_amp = np.median(wf_amp)
            std_amp = np.std(wf_amp)
            # remove outliers and recompute sta
            keep_idxs = np.where((med_amp - std_thresh * std_amp < wf_amp) & (wf_amp < med_amp + std_thresh * std_amp))[
                0]
            wf_keep = wf[keep_idxs]
            sta_clean = np.mean(wf_keep, 0)

            if plot_wf_hist:
                plt.figure()
                _ = plt.hist(wf_amp, bins=30)
                plt.axvline(med_amp - std_thresh * std_amp, color='r')
                plt.axvline(med_amp + std_thresh * std_amp, color='r')
        else:
            sta_clean = sta_raw

        # transpose template
        sta_clean = sta_clean.T

        np.save(template_save_path, sta_clean)
        np.save(locations_save_path, rec.get_channel_locations())

        t_stop = time.time()
        print(f"Elapsed time: {t_stop - t_start}s")

        return template_save_path, locations_save_path


def combine_multiple_configs(template_files, location_files):
    """
    Combine multiple templates on different configuraitons into a single template

    Parameters
    ----------
    template_files: list
        List of npy files containing templates for each configuration (num_channels, num_samples)
    location_files: list
        List of npy files containing locations for each configuration (num_channels, 2)

    Returns
    -------
    full_template: np.array
        The full templates on all locations (num_full_channels, num_samples)
    full_locations: np. array
        The full locations (num_full_channels, 2)
    """
    # load templates and locations
    templates = []
    locations = []
    for i, (template_file, loc_file) in enumerate(zip(template_files, location_files)):
        template = np.load(template_file)
        locs = np.load(loc_file)
        keep_idxs = []
        for t_idx, t in enumerate(template):
            if not np.all(np.diff(t) == 0):
                keep_idxs.append(t_idx)
        keep_idxs = np.array(keep_idxs)
        if len(keep_idxs) < len(template):
            print(f"Removed {len(template) - len(keep_idxs)} form template {i} for blank channels")
        templates.append(template[keep_idxs])
        locations.append(locs[keep_idxs])

    # find common locations and merge templates
    locations_set = [set(list([tuple(l) for l in locs])) for locs in locations]

    for l, lset in enumerate(locations_set):
        if l == 0:
            loc_union = lset
            loc_intersect = lset
        else:
            loc_union = loc_union | lset
            loc_intersect = loc_intersect & lset

    # TODO figure out how to use median instead!!!
    full_locations = np.array(list(loc_union))
    full_template = np.zeros((len(full_locations), templates[0].shape[1]))
    shared_channels = np.zeros(len(full_locations))

    locs_tuple = [tuple(loc) for loc in full_locations]
    for (temp, locs) in zip(templates, locations):
        # find location indexes
        loc_idxs = np.array([list(locs_tuple).index(tuple(loc)) for loc in locs])
        full_template[loc_idxs] += temp
        shared_channels[loc_idxs] += np.ones(len(loc_idxs), dtype='int')
    shared_channels[shared_channels == 0] = 1
    full_template = full_template / shared_channels[:, np.newaxis]

    return full_template, full_locations
