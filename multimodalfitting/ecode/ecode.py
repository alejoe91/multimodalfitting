import efel
import numpy as np
import pandas as pd
import time
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

import bluepyopt.ephys as ephys

from ..utils import calculate_eap
from .ecodestimuli import sAHP, HyperDepol, PosCheops, StimRecording, default_ecode_params


soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def generate_ecode_protocols(rheobase_current, delay_pre=250, delay_post=250,
                             record_extra=False, protocols_with_lfp=None, **stim_kwargs):
    """
    Generates E-CODE protocols given a certain rheobase current

    Parameters
    ----------
    rheobase_current: float
        Rheobase current in nA
    delay_pre: float
        Delay before stimulus onset in ms
    delay_post: float
        Delay after stimulus offset in ms
    record_extra: bool
        If True, LFPRecording is added to recordings
    protocols_with_lfp: list or None
        If None and record_extra is True, all protocols will record LFP.
        If list and record_extra is True, selected protocols will record LFP.
    stim_kwargs: dict
        Dictionary to update default_ecode_params dict

    Returns
    -------
    protocols: dict
        Dictionary with list of stimuli objects for each E-code protocol
    """
    ecode_stimuli = {}

    params = deepcopy(default_ecode_params)
    params.update(stim_kwargs)

    # 8 - IDthres
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

    ecode_stimuli["IDthres"] = stimuli

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
                        delay=delay_pre, tmid=delay_pre + duration_phase1,
                        tmid2=delay_pre + duration_phase1 + duration_phase2,
                        toff=delay_pre + duration_phase1 + duration_phase2 + duration_phase3,
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
            recordings = []
            recordings.append(ephys.recordings.CompRecording(name=f"{ecode_protocol_name}-{i_stim}.soma.v",
                                                             location=soma_loc, variable="v"))
            recordings.append(StimRecording(name=f"{ecode_protocol_name}-{i_stim}.stim.i"))
            if record_extra:
                if protocols_with_lfp is None:
                    recordings.append(ephys.recordings.LFPRecording(f"{ecode_protocol_name}-{i_stim}.MEA.LFP"))
                else:
                    if ecode_protocol_name in protocols_with_lfp:
                        recordings.append(ephys.recordings.LFPRecording(f"{ecode_protocol_name}-{i_stim}.MEA.LFP"))
            protocol = ephys.protocols.SweepProtocol(
                f"{ecode_protocol_name}-{i_stim}", [stimulus], recordings, cvode_active=True
            )
            protocols.append(protocol)
        ecode_protocols[ecode_protocol_name] = ephys.protocols.SequenceProtocol(name=ecode_protocol_name,
                                                                                protocols=protocols)
    return ecode_protocols


def compute_rheobase_for_model(cell, sim, step_duration=270, delay=250, step_min=0.1, step_max=1,
                               step_increment=0.02, isolate=True):
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
        response = protocol.run(cell_model=cell, param_values={}, sim=sim, isolate=isolate)
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
            print("Rheobase found")
            break

    if rheobase_current is None:
        print("Rheobase NOT found")

    return rheobase_current, protocols, responses


def run_ecode_protocols(protocols, cell, sim, resample_rate_khz=40):
    """
    Runs ecode protocols and resamples responses.

    Parameters
    ----------
    protocols: dict
        Dictionary of BluePyOpt protocols
    cell: LFPyCellModel
        BluePyOpt LFPy cell model
    sim: LFPySimulator
        BluePyOpt simulator
    resample_rate_khz: float
        Resample rate in kHz

    Returns
    -------
    resampled_responses: dict
        Dictionary with resampled responses for each protocol
    """
    ecode_responses = {}
    for i, (protocol_name, protocol) in enumerate(protocols.items()):
        ecode_responses[protocol_name] = {}
        print(f"Running protocol {protocol_name}")
        t_start = time.time()
        response = protocol.run(cell_model=cell, param_values={}, sim=sim)
        ecode_responses[protocol_name].update(response)
        print(f"Elapsed time {protocol_name}: {time.time() - t_start}")

    responses_interp_dict = dict()
    for protocol_name, responses in ecode_responses.items():
        responses_interp = dict()
        for response_name, response in responses.items():
            response_interp = _interpolate_response(response, resample_rate_khz)
            responses_interp[response_name] = response_interp
        responses_interp_dict[protocol_name] = responses_interp

    return responses_interp_dict


def save_intracellular_responses(responses_dict, output_folder):
    """
    Saves ecode responses to CSV

    Parameters
    ----------
    responses_dict: dict
        Dictionary with responses for each ecode (output of `run_ecode_protocols`)
    output_folder: str or Path
        Output folder to save ecode responses
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    for protocol_name, response in responses_dict.items():
        print(f"Saving {protocol_name}")
        (output_folder / protocol_name).mkdir(exist_ok=True, parents=True)
        dataframes_sweep = dict()
        for i, (resp_name, resp) in enumerate(response.items()):
            sweep = int(resp_name.split(".")[0].split('-')[1])
            if sweep not in dataframes_sweep.keys():
                dataframes_sweep[sweep] = pd.DataFrame()
            if "LFP" not in resp_name:
                for k, v in resp.items():
                    dataframes_sweep[sweep][k] = v
        for sweep, df in dataframes_sweep.items():
            df.to_csv(output_folder / protocol_name / f"{protocol_name}-{sweep}.csv")


def save_extracellular_template(responses, protocols, protocol_name,
                                probe, output_folder, resample_rate_khz=40,
                                sweep_id=0, **eap_kwargs):
    """
    Saves extracellular template, locations, and sampling_frequency in npy format in output_folder/'extracellular'.
    The sampling_frequency is saved in Hz.

    Parameters
    ----------
    responses: dict
        Dictionary with responses. It must contain the 'protocol-name'.MEA.LFP response and the
        associated 'protocol-name'.soma.v (to detect intracellular action potentials
    protocols: dict
        Dictionary of protocols
    protocol_name: str
        Protocol name to use to compute eap. If protocols have multiple sweeps, the protocol
        'protocole-name'-'sweep_id' is used.
    probe: MEAutility.MEA object
        The probe object
    output_folder: str or Path
        The output folder to save extracellular template, locations, and sampling_frequency
    resample_rate_khz: float
        Resampling rate in kHz (default=40)
    sweep_id: int
        In case of sweep protocols, the sweep id to use.
    eap_kwargs: kwargs
        Kwargs of the calculate_eap() function

    Returns
    -------
    eap: np.array (num_channels, num_samples)
        The extracellular action potential
    locations: np.array (num_channels, 2)
        The locations at which the eap is computed

    """
    if "fs" not in eap_kwargs.keys():
        eap_kwargs["fs"] = resample_rate_khz
    eap = calculate_eap(responses, protocol_name, protocols, sweep_id=sweep_id,
                        **eap_kwargs)
    locations = probe.positions

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    np.save(output_folder / "template.npy", eap)
    np.save(output_folder / "locations.npy", locations)
    np.save(output_folder / "fs.npy", resample_rate_khz * 1000)

    return eap, locations


def _interpolate_response(response, fs=20.0):
    """
    Helper function to reinterpolate responses
    """
    from scipy.interpolate import interp1d
    import pandas as pd

    x = response["time"]
    xnew = np.arange(np.min(x), np.max(x), 1.0 / fs)

    if isinstance(response.response, pd.DataFrame):
        other_columns = [k for k in list(response.response.columns) if k != "time"]
    else:
        other_columns = [k for k in list(response.response.keys()) if k != "time"]

    response_new = dict()
    response_new["time"] = xnew

    for other in other_columns:
        y = np.array(response[other])
        if y.ndim > 1:  # e.g. LFP responses
            f = interp1d(x, y, axis=1)
        else:
            f = interp1d(x, y)
        ynew = f(xnew)  # use interpolation function returned by `interp1d`
        response_new[other] = ynew

    return response_new
