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
        "tolerances": [10],
        "efeatures": ['mean_frequency'],
        "location": "soma"
    },
    "IV": {
        "amplitudes": [-100, -20, 0],  # -100 for the sag, -20 for the "passives", 0 for the RMP
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
        "amplitudes": [180, 260],  # Arbitrary choice
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
        "amplitudes": [-120],  # Arbitrary choice
        "tolerances": [10],
        "efeatures": {
            'Spikecount': [700, 970],
            'burst_number': [700, 970],
            'AP_amplitude': [700, 970],
            'ISI_values': [700, 970],
            'sag_amplitude': [250, 700],
            'sag_ratio1': [250, 700],
            'sag_ratio2': [250, 700],
        },
        "location": "soma"
    },
    "sAHP": {  # Used for validation, It's not obvious in Mikael's schema if the percentage is relative to the base or to the first step
        "amplitudes": [250],  # Arbitrary choice
        "tolerances": [10],
        "efeatures": {
            'Spikecount': [500, 725],
            'AP_amplitude': [500, 725],
            'ISI_values': [500, 725],
            'AHP_depth': [500, 725],
            'AHP_depth_abs': [500, 725],
            'AHP_time_from_peak': [500, 725],
            'steady_state_voltage_stimend': [500, 725]
        },
        "location": "soma"
    },
    "PosCheops": {  # Used for validation, need to check exact timings
        "amplitudes": [300],
        "tolerances": [10],
        "efeatures": ['Spikecount'],  # TODO: ISSUE HERE, CANNOT HAVE SEVERAL TIME THE SAME KEY
        "location": "soma"
    }
}

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

ecodes_timings = {
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
