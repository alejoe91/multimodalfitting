from ..ecode import HyperDepol, sAHP, PosCheops


def generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location):
    if isinstance(amplitudes, (int, float)):
        amplitudes = [amplitudes]
    if isinstance(tolerances, (int, float)):
        tolerances = [tolerances] * len(amplitudes)

    targets = []
    for amp, tol in zip(amplitudes, tolerances):
        for target in template:

            targets.append({
                'efeature': target['efeature'],
                'efel_settings': target['efel_settings'],
                'tolerance': amp,
                'amplitude': tol,
                'location': location,
                'protocol': protocol
            })

            if "efeature_name" in target:
                targets[-1]["efeature_name"] = target["efeature_name"]

    return targets


def get_idrest_targets(timings=None, stimulus=None):
    """IDrest targets: mean_frequency burst_number, adaptation_index2, ISI_CV,
    ISI_log_slope, inv_time_to_first_spike, inv_first_ISI, inv_second_ISI,
    inv_third_ISI, inv_fourth_ISI, inv_fifth_ISI, AP_amplitude, AHP_depth, AHP_time_from_peak,
    voltage_base, Spikecount"""

    protocol = 'IDrest'
    location = 'soma'
    amplitudes = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    tolerances = 20

    template = [
        {'efeature': 'voltage_base',
         'efel_settings': {}},
        {'efeature_name': 'Spikecount_pre_step',
         'efeature': 'Spikecount',
         'efel_settings': {'stim_start': 0., 'stim_end': timings['IDrest']['ton']}},
        {'efeature_name': 'Spikecount_post_step',
         'efeature': 'Spikecount',
         'efel_settings': {'stim_start': timings['IDrest']['toff'] + 100,
                           'stim_end': timings['IDrest']['tend']}},
        {'efeature': 'voltage_base',
         'efel_settings': {}},
        {'efeature': 'AHP_depth',
         'efel_settings': {}},
        {'efeature': 'AHP_time_from_peak',
         'efel_settings': {}},
        {'efeature': 'mean_frequency',
         'efel_settings': {}},
        {'efeature': 'burst_number',
         'efel_settings': {}},
        {'efeature': 'adaptation_index2',
         'efel_settings': {}},
        {'efeature  ': 'ISI_CV',
         'efel_settings': {}},
        {'efeature': 'ISI_log_slope',
         'efel_settings': {}},
        {'efeature': 'inv_time_to_first_spike',
         'efel_settings': {}},
        {'efeature': 'inv_first_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_second_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_third_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_fourth_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_fifth_ISI',
         'efel_settings': {}},
        {'efeature': 'AP_amplitude',
         'efel_settings': {}},
        {'efeature': 'AHP_depth',
         'efel_settings': {}},
        {'efeature': 'AHP_time_from_peak',
         'efel_settings': {}},
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)


def get_firepattern_targets(timings=None, stimulus=None):
    """firepattern targets: mean_frequency, burst_number, adaptation_index2, ISI_CV,
    ISI_log_slope, inv_time_to_first_spike, inv_first_ISI, inv_second_ISI,
    inv_third_ISI, inv_fourth_ISI, inv_fifth_ISI, AP_amplitude, AHP_depth, AHP_time_from_peak"""

    protocol = 'firepattern'
    location = 'soma'
    amplitudes = [120, 200]
    tolerances = 20

    template = [
        {'efeature': 'mean_frequency',
         'efel_settings': {}},
        {'efeature': 'burst_number',
         'efel_settings': {}},
        {'efeature': 'adaptation_index2',
         'efel_settings': {}},
        {'efeature': 'ISI_CV',
         'efel_settings': {}},
        {'efeature': 'ISI_log_slope',
         'efel_settings': {}},
        {'efeature': 'inv_time_to_first_spike',
         'efel_settings': {}},
        {'efeature': 'inv_first_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_second_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_third_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_fourth_ISI',
         'efel_settings': {}},
        {'efeature': 'inv_fifth_ISI',
         'efel_settings': {}},
        {'efeature': 'AP_amplitude',
         'efel_settings': {}},
        {'efeature': 'AHP_depth',
         'efel_settings': {}},
        {'efeature': 'AHP_time_from_peak',
         'efel_settings': {}}
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)


def get_iv_targets(timings=None, stimulus=None):
    """IV targets: Spikecount, voltage_base, voltage_deflection, voltage_deflection_begin,
    steady_state_voltage_stimend, ohmic_input_resistance_vb_ssse, sag_amplitude, sag_ratio1,
    sag_ratio2, decay_time_constant_after_stim"""

    protocol = 'IV'
    location = 'soma'
    amplitudes = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40, 60]
    tolerances = 10

    template = [
        {'efeature': 'Spikecount',
         'efel_settings': {}},
        {'efeature': 'voltage_base',
         'efel_settings': {}},
        {'efeature': 'voltage_deflection',
         'efel_settings': {}},
        {'efeature': 'voltage_deflection_begin',
         'efel_settings': {}},
        {'efeature': 'steady_state_voltage_stimend',
         'efel_settings': {}},
        {'efeature': 'ohmic_input_resistance_vb_ssse',
         'efel_settings': {}},
        {'efeature': 'sag_amplitude',
         'efel_settings': {}},
        {'efeature': 'sag_ratio1',
         'efel_settings': {}},
        {'efeature': 'sag_ratio2',
         'efel_settings': {}},
        {'efeature': 'decay_time_constant_after_stim',
         'efel_settings': {}},
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)


def get_apwaveform_targets(timings=None, stimulus=None):
    """APWaveform targets: AP_amplitude, AP1_amp, AP2_amp, AP_duration_half_width,
    AP_begin_width, AP_begin_voltage, AHP_depth, AHP_time_from_peak"""

    protocol = 'APWaveform'
    location = 'soma'
    amplitudes = [200, 230, 260, 290, 320, 350]
    tolerances = 20

    template = [
        {'efeature': 'AP_amplitude',
         'efel_settings': {}},
        {'efeature': 'AP1_amp',
         'efel_settings': {}},
        {'efeature': 'AP2_amp',
         'efel_settings': {}},
        {'efeature': 'AP_duration_half_width',
         'efel_settings': {}},
        {'efeature': 'AP_begin_width',
         'efel_settings': {}},
        {'efeature': 'AP_begin_voltage',
         'efel_settings': {}},
        {'efeature': 'AHP_depth',
         'efel_settings': {}},
        {'efeature': 'AHP_time_from_peak',
         'efel_settings': {}},
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)


def get_hyperdepol_targets(timings=None, stimulus=None):
    """HyperDepol targets:
    depol: Spikecount, burst_number, AP_amplitude, ISI_values
    hyper: sag_amplitude, sag_ratio1, sag_ratio2
    """

    protocol = 'HyperDepol'
    location = 'soma'
    amplitudes = [-160, -120, -80, -40]
    tolerances = 20

    if timings is not None:
        ton = timings["HyperDepol"]["ton"]
        tmid = timings["HyperDepol"]["tmid"]
        toff = timings["HyperDepol"]["toff"]
    else:
        assert stimulus is not None
        assert isinstance(stimulus, HyperDepol)
        ton = stimulus.delay
        tmid = stimulus.tmid
        toff = stimulus.toff

    template = [
        {'efeature': 'Spikecount',
         'efel_settings': {'stim_start': tmid, 'stim_end': toff}},
        {'efeature': 'burst_number',
         'efel_settings': {'stim_start': tmid, 'stim_end': toff}},
        {'efeature': 'AP_amplitude',
         'efel_settings': {'stim_start': tmid, 'stim_end': toff}},
        {'efeature': 'ISI_values',
         'efel_settings': {'stim_start': tmid, 'stim_end': toff}},
        {'efeature': 'sag_amplitude',
         'efel_settings': {'stim_start': ton, 'stim_end': tmid}},
        {'efeature': 'sag_ratio1',
         'efel_settings': {'stim_start': ton, 'stim_end': tmid}},
        {'efeature': 'sag_ratio2',
         'efel_settings': {'stim_start': ton, 'stim_end': tmid}},
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)


def get_sahp_targets(timings=None, stimulus=None):
    """sAHP targets: Spikecount, AP_amplitude, ISI_values, AHP_depth, AHP_depth_abs,
    AHP_time_from_peak, steady_state_voltage_stimend
    """

    protocol = 'sAHP'
    location = 'soma'
    amplitudes = [150, 200, 250, 300]
    tolerances = 10

    if timings is not None:
        tmid = timings["sAHP"]["tmid"]
        tmid2 = timings["sAHP"]["tmid2"]
    else:
        assert stimulus is not None
        assert isinstance(stimulus, sAHP)
        tmid = stimulus.tmid
        tmid2 = stimulus.tmid2

    template = [
        {'efeature': 'Spikecount',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}},
        {'efeature': 'AP_amplitude',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}},
        {'efeature': 'ISI_values',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}},
        {'efeature': 'AHP_depth',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}},
        {'efeature': 'AHP_depth_abs',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}},
        {'efeature': 'AHP_time_from_peak',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}},
        {'efeature': 'steady_state_voltage_stimend',
         'efel_settings': {'stim_start': tmid, 'stim_end': tmid2}}
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)


def get_poscheops_targets(timings=None, stimulus=None):
    """PosCheops targets: Spikecount, mean_frequency, burst_number, adaptation_index2,
    ISI_CV, ISI_log_slope, inv_time_to_first_spike, inv_first_ISI, inv_second_ISI,
    inv_third_ISI, inv_fourth_ISI, inv_fifth_ISI"""

    protocol = 'PosCheops'
    location = 'soma'
    amplitudes = [300]
    tolerances = 10

    if timings is not None:
        ton = timings["PosCheops"]["ton"]
        t1 = timings["PosCheops"]["t1"]
        t2 = timings["PosCheops"]["t2"]
        t3 = timings["PosCheops"]["t3"]
        t4 = timings["PosCheops"]["t4"]
        toff = timings["PosCheops"]["toff"]
    else:
        assert stimulus is not None
        assert isinstance(stimulus, PosCheops)
        ton = stimulus.delay
        t1 = stimulus.t1
        t2 = stimulus.t2
        t3 = stimulus.t3
        t4 = stimulus.t4
        toff = stimulus.toff

    template = [
        {'efeature_name': 'Spikecount_p1',
         'efeature': 'Spikecount',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'mean_frequency_p1',
         'efeature': 'mean_frequency',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'burst_number_p1',
         'efeature': 'burst_number',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'adaptation_index2_p1',
         'efeature': 'adaptation_index2',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'ISI_CV_p1',
         'efeature': 'ISI_CV',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'ISI_log_slope_p1',
         'efeature': 'ISI_log_slope',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'inv_time_to_first_spike_p1',
         'efeature': 'inv_time_to_first_spike',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'inv_first_ISI_p1',
         'efeature': 'inv_first_ISI',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'inv_second_ISI_p1',
         'efeature': 'inv_second_ISI',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'inv_third_ISI_p1',
         'efeature': 'inv_third_ISI',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'inv_fourth_ISI_p1',
         'efeature': 'inv_fourth_ISI',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'inv_fifth_ISI_p1',
         'efeature': 'inv_fifth_ISI',
         'efel_settings': {'stim_start': ton, 'stim_end': t1}},
        {'efeature_name': 'Spikecount_p2',
         'efeature': 'Spikecount',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'mean_frequency_p2',
         'efeature': 'mean_frequency',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'burst_number_p2',
         'efeature': 'burst_number',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'adaptation_index2_p2',
         'efeature': 'adaptation_index2',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'ISI_CV_p2',
         'efeature': 'ISI_CV',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'ISI_log_slope_p2',
         'efeature': 'ISI_log_slope',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'inv_time_to_first_spike_p2',
         'efeature': 'inv_time_to_first_spike',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'inv_first_ISI_p2',
         'efeature': 'inv_first_ISI',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'inv_second_ISI_p2',
         'efeature': 'inv_second_ISI',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'inv_third_ISI_p2',
         'efeature': 'inv_third_ISI',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'inv_fourth_ISI_p2',
         'efeature': 'inv_fourth_ISI',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'inv_fifth_ISI_p2',
         'efeature': 'inv_fifth_ISI',
         'efel_settings': {'stim_start': t2, 'stim_end': t3}},
        {'efeature_name': 'Spikecount_p3',
         'efeature': 'Spikecount',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'mean_frequency_p3',
         'efeature': 'mean_frequency',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'burst_number_p3',
         'efeature': 'burst_number',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'adaptation_index2_p3',
         'efeature': 'adaptation_index2',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'ISI_CV_p3',
         'efeature': 'ISI_CV',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'ISI_log_slope_p3',
         'efeature': 'ISI_log_slope',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'inv_time_to_first_spike_p3',
         'efeature': 'inv_time_to_first_spike',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'inv_first_ISI_p3',
         'efeature': 'inv_first_ISI',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'inv_second_ISI_p3',
         'efeature': 'inv_second_ISI',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'inv_third_ISI_p3',
         'efeature': 'inv_third_ISI',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'inv_fourth_ISI_p3',
         'efeature': 'inv_fourth_ISI',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}},
        {'efeature_name': 'inv_fifth_ISI_p3',
         'efeature': 'inv_fifth_ISI',
         'efel_settings': {'stim_start': t4, 'stim_end': toff}}
    ]

    return generate_targets_at_amplitudes(template, amplitudes, tolerances, protocol, location)
