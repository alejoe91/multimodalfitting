from ecodestimuli import sAHP, HyperDepol, PosCheops
import bluepyopt.ephys as ephys
import numpy as np
from copy import deepcopy

# define default params

_default_params = {
    "IDthres": {
        'duration': 270,
        'from': 0.5,
        'to': 1.3,
        'step': 0.04,
    },
    "firepattern": {
        'duration': 3600,
        'from': 1.2,
        'to': 2,
        'step': 0.8,
    },
    "IV": {
        'duration': 3000,
        'from': -1.4,
        'to': 0.6,
        'step': 0.2,
    },
    "IDrest": {
        'duration': 1350,
        'from': 0.5,
        'to': 3,
        'step': 0.25,
    },
    "APWaveform": {
        'duration': 50,
        'from': 2,
        'to': 3,
        'step': 0.25,
    },
    "HyperDepol": {
        'hyper_duration': 450,
        'hyper_from': -0.4,
        'hyper_to': -1.6,
        'hyper_step': -0.4,
        'depol_duration': 270,
        'depol_amp': 1
    },
    "sAHP": {
        'phase1_duration': 250,
        'phase1_amp': 0.4,
        'phase2_duration': 225,
        'phase2_from': 1.5,
        'phase2_to': 3,
        'phase2_step': 0.5,
        'phase3_duration': 450,
        'phase3_amp': 0.4
    },
    "PosCheops": {
        'amp': 3,
        'duration1': 4000,
        'duration2': 2,
        'duration3': 1.33,
        'delay': 1500,
    },
}

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def generate_ecode_stimuli(rheobase_current, delay_pre=250, delay_post=250,
                           **stim_kwargs):
    """
    Generates E-CODE protocols given a certain rheobase current

    Parameters
    ----------
    rheobase_current: float
        Rheobase current in pA
    stim_kwargs: dict
        Dictionary to update _default_params dict

    Returns
    -------
    protocols: dict
        Dictionary with list of stimuli objects for each Ecode protocol
    """
    ecode_stimuli = {}

    params = deepcopy(_default_params)
    params.update(stim_kwargs)

    # 8 - IDthresh
    steps = np.arange(params["IDthres"]["from"], params["IDthres"]["to"] + 0.001, params["IDthres"]["step"])
    amplitude_steps = steps * rheobase_current
    duration = params["IDthres"]["duration"]

    stimuli = []
    for amp in amplitude_steps:
        stimulus = ephys.stimuli.LFPySquarePulse(step_amplitude=amp, step_delay=delay_pre,
                                                 step_duration=duration,
                                                 total_duration=delay_pre + duration + delay_post,
                                                 location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["IDthresh"] = stimuli

    # 9 - firepattern
    steps = np.arange(params["firepattern"]["from"], params["firepattern"]["to"] + 0.001, params["firepattern"]["step"])
    amplitude_steps = steps * rheobase_current
    duration = params["firepattern"]["duration"]

    stimuli = []
    for amp in amplitude_steps:
        stimulus = ephys.stimuli.LFPySquarePulse(step_amplitude=amp, step_delay=delay_pre,
                                                 step_duration=duration,
                                                 total_duration=delay_pre + duration + delay_post,
                                                 location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["firepattern"] = stimuli

    # 10 - IV
    steps = np.arange(params["IV"]["from"], params["IV"]["to"] + 0.001, params["IV"]["step"])
    amplitude_steps = steps * rheobase_current
    duration = params["IV"]["duration"]

    stimuli = []
    for amp in amplitude_steps:
        stimulus = ephys.stimuli.LFPySquarePulse(step_amplitude=amp, step_delay=delay_pre,
                                                 step_duration=duration,
                                                 total_duration=delay_pre + duration + delay_post,
                                                 location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["IV"] = stimuli

    # 11 - IDrest
    steps = np.arange(params["IDrest"]["from"], params["IDrest"]["to"] + 0.001, params["IDrest"]["step"])
    amplitude_steps = steps * rheobase_current
    duration = params["IDrest"]["duration"]

    stimuli = []
    for amp in amplitude_steps:
        stimulus = ephys.stimuli.LFPySquarePulse(step_amplitude=amp, step_delay=delay_pre,
                                                 step_duration=duration,
                                                 total_duration=delay_pre + duration + delay_post,
                                                 location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["IDrest"] = stimuli

    # 12 - APWaveform
    steps = np.arange(params["APWaveform"]["from"], params["APWaveform"]["to"] + 0.001, params["APWaveform"]["step"])
    amplitude_steps = steps * rheobase_current
    duration = params["APWaveform"]["duration"]

    stimuli = []
    for amp in amplitude_steps:
        stimulus = ephys.stimuli.LFPySquarePulse(step_amplitude=amp, step_delay=delay_pre,
                                                 step_duration=duration,
                                                 total_duration=delay_pre + duration + delay_post,
                                                 location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["APWaveform"] = stimuli

    # 13 - HyperDepol
    steps = np.arange(params["HyperDepol"]["hyper_from"], params["HyperDepol"]["hyper_to"] - 0.001,
                      params["HyperDepol"]["hyper_step"])
    amplitude_steps = steps * rheobase_current
    duration_hyper = params["HyperDepol"]["hyper_duration"]
    amp_depol = params["HyperDepol"]["depol_amp"] * rheobase_current
    duration_depol = params["HyperDepol"]["depol_duration"]

    stimuli = []
    for amp in amplitude_steps:
        stimulus = HyperDepol(hyperpol_amplitude=amp, depol_amplitude=amp_depol,
                              delay=delay_pre, tmid=delay_pre + duration_hyper,
                              toff=delay_pre + duration_hyper + duration_depol,
                              total_duration=delay_pre + duration_hyper + duration_depol + delay_post,
                              location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["HyperDepol"] = stimuli

    # 14 - sAHP
    steps = np.arange(params["sAHP"]["phase2_from"], params["sAHP"]["phase2_to"] + 0.001,
                      params["sAHP"]["phase2_step"])
    amplitude_steps = steps * rheobase_current
    duration_phase2 = params["sAHP"]["phase2_duration"]
    duration_phase1 = params["sAHP"]["phase1_duration"]
    amp_phase1 = params["sAHP"]["phase1_amp"] * rheobase_current
    duration_phase3 = params["sAHP"]["phase3_duration"]
    amp_phase3 = params["sAHP"]["phase3_amp"] * rheobase_current

    stimuli = []
    for amp in amplitude_steps:
        stimulus = sAHP(phase1_amplitude=amp_phase1, phase3_amplitude=amp_phase3, phase2_amplitude=amp,
                        delay=delay_pre, sahp_tmid=delay_pre + duration_phase1,
                        sahp_tmid2=delay_pre + duration_phase1 + duration_phase2,
                        sahp_toff=delay_pre + duration_phase1 + duration_phase2 + duration_phase3,
                        total_duration=delay_pre + duration_phase1 + duration_phase2 + duration_phase3 + delay_post,
                        location=soma_loc)
        stimuli.append(stimulus)

    ecode_stimuli["sAHP"] = stimuli

    # 15 - PosCheops

    amp = params["PosCheops"]["amp"] * rheobase_current
    duration1 = params["PosCheops"]["duration1"]
    duration2 = params["PosCheops"]["duration2"]
    duration3 = params["PosCheops"]["duration3"]
    interramp_delay = params["PosCheops"]["delay"]

    stimulus = PosCheops(ramp1_dur=duration1, ramp2_dur=duration2, ramp3_dur=duration3,
                         ramp1_amp=amp, ramp2_amp=amp, ramp3_amp=amp,
                         ramp12_delay=interramp_delay, ramp23_delay=interramp_delay,
                         total_duration=delay_pre + 2 * duration1 + 2 * duration2 + 2 * duration3 +
                                        2 * interramp_delay + delay_post)
    ecode_stimuli["PosCheops"] = [stimulus]

    return ecode_stimuli


def generate_ecode_stimuli_from_data(wcp_files, delay_pre=250, delay_post=250,
                                     **stim_kwargs):

    import neo

    pass
