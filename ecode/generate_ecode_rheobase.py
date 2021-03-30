from ecodestimuli import sAHP, HyperDepol, PosCheops
import bluepyopt.ephys as ephys
import numpy as np
from copy import deepcopy
import efel
from collections import OrderedDict

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
        'duration2': 2000,
        'duration3': 1330,
        'delay': 1500,
    },
}

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def generate_ecode_protocols(rheobase_current, delay_pre=250, delay_post=250,
                             feature_set="soma", **stim_kwargs):
    """
    Generates E-CODE protocols given a certain rheobase current

    Parameters
    ----------
    rheobase_current: float
        Rheobase current in nA
    stim_kwargs: dict
        Dictionary to update _default_params dict

    Returns
    -------
    protocols: dict
        Dictionary with list of stimuli objects for each E-code protocol
    """
    ecode_stimuli = {}

    params = deepcopy(_default_params)
    params.update(stim_kwargs)

    # 8 - IDthresh
    steps = np.arange(params["IDthres"]["from"], params["IDthres"]["to"] + 0.0001, params["IDthres"]["step"])
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
    steps = np.arange(params["firepattern"]["from"], params["firepattern"]["to"] + 0.0001,
                      params["firepattern"]["step"])
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
    steps = np.arange(params["IV"]["from"], params["IV"]["to"] + 0.0001, params["IV"]["step"])
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
    steps = np.arange(params["IDrest"]["from"], params["IDrest"]["to"] + 0.0001, params["IDrest"]["step"])
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
    steps = np.arange(params["APWaveform"]["from"], params["APWaveform"]["to"] + 0.0001, params["APWaveform"]["step"])
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
    steps = np.arange(params["HyperDepol"]["hyper_from"], params["HyperDepol"]["hyper_to"] - 0.0001,
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
    steps = np.arange(params["sAHP"]["phase2_from"], params["sAHP"]["phase2_to"] + 0.0001,
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

    total_duration = delay_pre + 2 * duration1 + 2 * duration2 + 2 * duration3 + 2 * interramp_delay + delay_post

    stimulus = PosCheops(ramp1_dur=duration1, ramp2_dur=duration2, ramp3_dur=duration3,
                         ramp1_amp=amp, ramp2_amp=amp, ramp3_amp=amp,
                         ramp12_delay=interramp_delay, ramp23_delay=interramp_delay,
                         total_duration=total_duration,
                         location=soma_loc)
    ecode_stimuli["PosCheops"] = [stimulus]

    ecode_protocols = OrderedDict()
    for ecode_protocol_name, stimuli in ecode_stimuli.items():
        # define somatic recording
        protocols = []
        for i_stim, stimulus in enumerate(stimuli):
            recordings = [
                ephys.recordings.CompRecording(
                    name=f"{ecode_protocol_name}-{i_stim}.soma.v", location=soma_loc, variable="v"
                )
            ]
            if feature_set == "extra":
                recordings.append(ephys.recordings.LFPRecording(f"{ecode_protocol_name}-{i_stim}.MEA.LFP"))
            protocol = ephys.protocols.SweepProtocol(
                f"{ecode_protocol_name}-{i_stim}", [stimulus], recordings, cvode_active=True
            )
            protocols.append(protocol)
        ecode_protocols[ecode_protocol_name] = ephys.protocols.SequenceProtocol(name=ecode_protocol_name,
                                                                                protocols=protocols)

    return ecode_protocols


def compute_rheobase_for_model(cell, sim, step_duration=270, delay=250, step_min=0.1, step_max=1,
                               step_increment=0.02):
    """
    
    Parameters
    ----------
    cell: BluePyOpt.ephys.Cell
        The BluePyOpt cell model 
    step_duration: float
        Step duration in ms
    delay: float
        Delay in ms before and after the step
    step_min: float
        Minimum step current in nA
    step_max: float
        Maximum step current in nA
    step_increment: float
        Step increment in nA

    Returns
    -------
    rheobase_current: float or None
        The estimated rheobase current in nA. If no spikes are found, it returns None
    """
    protocol_name = "StepsRheobase"
    step_currents = np.arange(step_min, step_max + 0.001, step_increment)

    # define stimulus
    protocols = []

    # define somatic recording
    recordings = [
        ephys.recordings.CompRecording(
            name="soma.v", location=soma_loc, variable="v"
        )
    ]

    for curr in step_currents:
        stimulus = ephys.stimuli.LFPySquarePulse(
            step_amplitude=curr,
            step_delay=delay,
            step_duration=step_duration,
            location=soma_loc,
            total_duration=2 * delay + step_duration)
        protocol = ephys.protocols.SweepProtocol(
            f"{protocol_name}_{np.round(curr, 3)}nA", [stimulus], recordings, cvode_active=True
        )
        protocols.append(protocol)

    rheobase_current = None
    responses = []
    for i, protocol in enumerate(protocols):
        print(f"Running protocol {protocol.name}")
        response = protocol.run(cell_model=cell, param_values={}, sim=sim)
        responses.append(response)

        # make efel object
        trace = {}
        trace["T"] = response["soma.v"]["time"].values
        trace["V"] = response["soma.v"]["voltage"].values
        trace['stim_start'] = [delay]
        trace['stim_end'] = [delay + step_duration]

        features = efel.getFeatureValues([trace], ['Spikecount'])
        spikecount = features[0]["Spikecount"][0]
        print(f"Number of spikes: {spikecount}")

        if spikecount == 1:
            rheobase_current = step_currents[i]
            print(f"Rheobase found")
            break

    if rheobase_current is None:
        print(f"Rheobase NOT found")

    return rheobase_current, protocols, responses


def generate_ecode_stimuli_from_data(wcp_files, delay_pre=250, delay_post=250,
                                     **stim_kwargs):
    import neo

    pass
