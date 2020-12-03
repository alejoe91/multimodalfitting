"""Run simple cell optimisation"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
import itertools
import model

import bluepyopt as bpopt
import bluepyopt.ephys as ephys

import time
import logging
import LFPy
import model

from utils import _filter_response, _interpolate_response, _upsample_wf, _get_peak_times, _get_waveforms

logger = logging.getLogger("__main__")

script_dir = os.path.dirname(__file__)
config_dir = os.path.join(script_dir, "config")

soma_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)


def get_protocol_definitions():
    return json.load(open(os.path.join(config_dir, "protocols.json")))


def get_feature_definitions(feature_set=None, feature_file=None):

    if feature_file is not None:
        assert feature_set is not None
        return pickle.load(open(feature_file, 'rb'))
    else:
        return json.load(open(os.path.join(config_dir, 'features.json')))['multiple']


def define_recordings(protocol_name, protocol_definition, electrode=None):
    
    recordings = [
        ephys.recordings.CompRecording(
            name="%s.soma.v" % protocol_name, location=soma_loc, variable="v"
        )
    ]
    
    if "extra_recordings" in protocol_definition:

            for recording_definition in protocol_definition["extra_recordings"]:
                
                if recording_definition["type"] == "somadistance":
                    
                    location = ephys.locations.NrnSomaDistanceCompLocation(
                        name=recording_definition["name"],
                        soma_distance=recording_definition["somadistance"],
                        seclist_name=recording_definition["seclist_name"],
                    )
                    
                    var = recording_definition["var"]
                    recordings.append(
                            ephys.recordings.CompRecording(
                            name="%s.%s.%s" % (protocol_name, location.name, var),
                            location=location,
                            variable=recording_definition["var"],
                        )
                    )

    if electrode is not None:
        recordings.append(ephys.recordings.LFPRecording("%s.MEA.LFP" % protocol_name))

    return recordings


def define_stimuli(protocol_name, protocol_definition):

    stimuli = []
    for stimulus_definition in protocol_definition["stimuli"]:

        if protocol_name in ['BAC', 'Step1', 'Step2', 'Step3', 'bAP']:

            stimuli.append(ephys.stimuli.LFPySquarePulse(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=soma_loc,
                total_duration=stimulus_definition['totduration']))

        if protocol_name in ['EPSP', 'BAC']:

            loc_api = ephys.locations.NrnSomaDistanceCompLocation(
                        name=stimulus_definition["name"],
                        soma_distance=620,
                        seclist_name="apical",
                    )

            stimuli.append(ephys.stimuli.LFPySquarePulse(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=loc_api,
                total_duration=stimulus_definition['totduration']))

        if protocol_name in ['CaBurst']:

            loc_api = ephys.locations.NrnSomaDistanceCompLocation(
                        name=stimulus_definition["name"],
                        soma_distance=620,
                        seclist_name="apical",
                    )

            stimuli.append(ephys.stimuli.LFPySquarePulse(
                step_amplitude=stimulus_definition['amp'],
                step_delay=stimulus_definition['delay'],
                step_duration=stimulus_definition['duration'],
                location=loc_api,
                total_duration=stimulus_definition['totduration']))
    
    return stimuli


def define_protocols(feature_set=None, feature_file=None, electrode=None):

    protocol_definitions = get_protocol_definitions()
    feature_definitions = get_feature_definitions(feature_set, feature_file)
    protocols = {}

    for protocol_name in feature_definitions:

        recordings = define_recordings(protocol_name, protocol_definitions[protocol_name], electrode)
        stimuli = define_stimuli(protocol_name, protocol_definitions[protocol_name])

        protocols[protocol_name] = ephys.protocols.SweepProtocol(
            protocol_name, stimuli, recordings, cvode_active=True
        )
        
    return protocols


def get_release_params():
    # load release params
    release_params_file = os.path.join(config_dir, "parameters_release.json")
    # load unfrozen params
    params_file = os.path.join(config_dir, "parameters.json")

    all_release_params = {}
    with open(release_params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            all_release_params[f"{prm['param_name']}.{prm['sectionlist']}"] = prm["value"]

    params_bounds = {}
    with open(params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = prm["bounds"]

    release_params = {}
    for k, v in all_release_params.items():
        if k in params_bounds.keys():
            release_params[k] = v

    return release_params


def get_unfrozen_params_bounds():
    # load unfrozen params
    params_file = os.path.join(config_dir, "parameters.json")

    params_bounds = {}
    with open(params_file, 'r') as f:
        data = json.load(f)

        for prm in data:
            if "bounds" in prm:
                params_bounds[f"{prm['param_name']}.{prm['sectionlist']}"] = prm["bounds"]

    return params_bounds


def compute_feature_values(params, cell_model, protocols, sim, feature_set='bap', std=0.2, 
                           probe=None, channels=None, detect_threshold=0, verbose=False):
    """
    Calculate features for cell model and protocols.
    Parameters
    ----------
    params
    cell_model
    protocols
    sim
    feature_set
    std
    feature_folder
    probe
    channels: list or None
    save_to_file
    Returns
    -------
    """
    assert feature_set in ['bap', 'soma', 'extra', 'all']

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
                        if channels is not 'map':
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
            print(time.time()-t1)
        
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
                if channels is not 'map':
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


def calculate_eap(responses, protocol_name, protocols, fs=20, fcut=1,
                  ms_cut=[2, 10], upsample=10, skip_first_spike=True, skip_last_spike=True,
                  raise_warnings=False, verbose=False, **efel_kwargs):
    """
    Calculate extracellular action potential (EAP)

    Parameters
    ----------
    responses
    protocol_name
    protocols
    fs
    fcut
    ms_cut
    upsample
    skip_first_spike
    skip_last_spike
    raise_warnings
    verbose
    efel_kwargs

    Returns
    -------

    """
    assert "Step" in protocol_name
    stimulus = protocols[protocol_name].stimuli[0]
    stim_start = stimulus.step_delay
    stim_end = stimulus.step_delay + stimulus.step_duration
    efel_kwargs['threshold'] = -20

    somatic_recording_name = f'{protocol_name}.soma.v'
    extra_recording_name = f'{protocol_name}.MEA.LFP'

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
        response_filter = _filter_response(response_interp, fcut=fcut)
    else:
        if verbose:
            print('filter disabled')
        response_filter = response_interp

    ewf = _get_waveforms(response_filter, peak_times, ms_cut)
    mean_wf = np.mean(ewf, axis=0)
    if upsample is not None:
        if verbose:
            print('upsample')
        assert upsample > 0
        upsample = int(upsample)
        mean_wf_up = _upsample_wf(mean_wf, upsample)
        fs_up = upsample * fs
    else:
        mean_wf_up = mean_wf
        fs_up = fs

    return mean_wf_up


def define_fitness_calculator(protocols, feature_file, feature_set, channels=None, probe=None):
    """Define fitness calculator"""
    
    assert feature_set in ['multiple', 'soma', 'extra']
    if feature_set == 'extra':
        assert probe is not None, "Provide a MEAutility probe to use the 'extra' set"

    feature_definitions = get_feature_definitions(feature_set, feature_file)
    
    objectives = []
    efeatures = {}
    for protocol_name, locations in feature_definitions.items():
        
        efeatures[protocol_name] = []
        
        for location, features in locations.items():
            
            for efel_feature_name, meanstd in features.items():
                
                feature_name = '%s.%s.%s' % (protocol_name, location, efel_feature_name)
                
                stimulus = protocols[protocol_name].stimuli[0]

                kwargs = {'stim_start': stimulus.step_delay}

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
                    
                    if channels == 'map':
                        kwargs['channel_id'] = None
                        kwargs['extrafel_feature_name'] = efel_feature_name
                        
                    else:
                        channel_id = int(efel_feature_name.split('_')[-1])
                        kwargs['extrafel_feature_name'] = '_'.join(efel_feature_name.split('_')[:-1])
                        if channels is not None and channel_id not in channels:
                            continue
                        else:
                            kwargs['channel_id'] = channel_id

                else:
                    feature_class = ephys.efeatures.eFELFeature
                    kwargs['efel_feature_name'] = efel_feature_name
                    kwargs['recording_names'] = {'': '%s.%s.v' % (protocol_name, location)}

                feature = feature_class(
                    feature_name,
                    exp_mean=meanstd[0],
                    exp_std=meanstd[1],
                    **kwargs
                )
                
                efeatures[protocol_name].append(feature)
                
                objectives.append(
                    ephys.objectives.SingletonObjective(
                        feature_name,
                        feature
                    )
                )

    return ephys.objectivescalculators.ObjectivesCalculator(objectives), efeatures


def prepare_optimization(feature_set, sample_id, offspring_size=10, config_path='config',
                         channels=None, map_function=None, probe_type='linear', seed=1, morph_modifier=""):
    
    config_path = Path(config_path)
    feature_file = config_path / 'features' / f'random_{sample_id}' / f'{feature_set}.pkl'
    
    probe = None
    electrode = None
    
    if feature_set == "extra" and channels == 'map':
        assert probe_type in ['linear', 'planar']
        probe_file = config_path / 'features' / f'random_{sample_id}' / 'probe.json'
        probe, electrode = model.define_electrode(probe_file=probe_file)

    cell = model.create(morph_modifier, release=False)

    param_names = [param.name for param in cell.params.values() if not param.frozen]

    fitness_protocols = define_protocols(feature_set, feature_file, electrode=electrode)
    
    sim = ephys.simulators.LFPySimulator(LFPyCellModel=cell, cvode_active=True, electrode=electrode)

    fitness_calculator, _ = define_fitness_calculator(
        protocols=fitness_protocols,
        feature_file=feature_file,
        feature_set=feature_set,
        probe=probe, 
        channels=channels
    )

    evaluator = ephys.evaluators.CellEvaluator(cell_model=cell,
                                               param_names=param_names,
                                               fitness_protocols=fitness_protocols,
                                               fitness_calculator=fitness_calculator,
                                               sim=sim,
                                               timeout=900)

    opt = bpopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
            evaluator=evaluator,
            offspring_size=20,
            seed=seed,
            map_function=map_function,
            weight_hv=0.4,
            selector_name="multi_objective"
    )

    return {'optimisation': opt, 'evaluator': evaluator, 'objectives_calculator': fitness_calculator,
            'protocols': fitness_protocols}


def run_optimization(feature_set, sample_id, opt, channels, max_ngen, seed=1, prob_type=None):
    
    if channels is None:
        nchannels = 'all'
    elif channels is 'map':
        nchannels = f'map-{prob_type}'
    else:
        nchannels = len(channels)
    
    cp_filename = Path('checkpoints') / f'random_{sample_id}' / f'{feature_set}_off{opt.offspring_size}_' \
                                                                f'ngen{max_ngen}_{nchannels}chan_{seed}seed.pkl'
    
    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False
    
    final_pop, halloffame, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_filename, continue_cp=continue_cp)
    
    return {'final_pop': final_pop, 'halloffame': halloffame, 'log': log, 'hist': hist}
