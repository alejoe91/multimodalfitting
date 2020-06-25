# 3) Run optimization
#
# In this notebook we use the previously computed features to fit the cell model parameters and save the results.
#
# We can run optimization on a subset of samples/feature sets.

import bluepyopt as bpopt
import bluepyopt.ephys as ephys

import matplotlib
import matplotlib.pyplot as plt
import MEAutility as mu
import json
import numpy
import time
import numpy as np
import LFPy
from pathlib import Path
import pandas as pd
import os
from datetime import datetime

matplotlib.use('Agg')

import l5pc_model
import l5pc_evaluator
import l5pc_plot

from ipyparallel import Client

save_fig = True

# Define feature sets and sample id to fit
feature_sets = ['extra']  # ['soma', 'bap', 'extra']
sample_ids = [3]  # [0, ..., n_samples]

# Define optimization parameters
offspring_size = 250
max_ngen = 50

# (optional) define a subset of channels to use (if None all channels are used)
channels = [0, 6, 7, 10, 15]
n_channels = len(channels)

print(f"Starting optimizations on random samples {sample_ids} - feature sets {feature_sets} "
      f"- max_ngen {max_ngen} - pop size {offspring_size} - channels {channels}")


def prepare_optimization(feature_set, sample_id, offspring_size=10, config_path='config',
                         channels=None, map_function=None):
    config_path = Path(config_path)
    morphology = ephys.morphologies.NrnFileMorphology('morphology/C060114A7.asc', do_replace_axon=True)
    parameters = l5pc_model.define_parameters()
    mechanisms = l5pc_model.define_mechanisms()

    cell = ephys.models.LFPyCellModel('l5pc', v_init=-65., morph=morphology, mechs=mechanisms, params=parameters)

    param_names = [param.name for param in cell.params.values() if not param.frozen]

    if feature_set == "extra":
        probe_file = config_path / 'features' / f'random_{sample_id}' / 'probe.json'
        with probe_file.open('r') as f:
            info = json.load(f)
        probe = mu.return_mea(info=info)
        electrode = LFPy.RecExtElectrode(probe=probe)
        if channels is None:
            print(f"MEA z positions:\n{probe.positions[:, 1]}")

        else:
            print(f"MEA z positions:\n{probe.positions[channels, 1]}")
    else:
        probe = None
        electrode = None

    fitness_protocols = l5pc_evaluator.define_protocols(electrode=electrode)
    sim = ephys.simulators.LFPySimulator(LFPyCellModel=cell, cvode_active=True, electrode=electrode)

    feature_file = config_path / 'features' / f'random_{sample_id}' / f'{feature_set}.json'
    fitness_calculator = l5pc_evaluator.define_fitness_calculator(protocols=fitness_protocols,
                                                                  feature_file=feature_file,
                                                                  probe=probe, channels=channels)

    print(f'Number of features: {len(fitness_calculator.objectives)}')

    evaluator = ephys.evaluators.CellEvaluator(cell_model=cell,
                                               param_names=param_names,
                                               fitness_protocols=fitness_protocols,
                                               fitness_calculator=fitness_calculator,
                                               sim=sim)

    opt = bpopt.optimisations.DEAPOptimisation(evaluator=evaluator,
                                               offspring_size=offspring_size,
                                               map_function=map_function)

    output = {'optimisation': opt, 'evaluator': evaluator, 'objectives_calculator': fitness_calculator,
              'protocols': fitness_protocols}

    return output


def run_optimization(feature_set, sample_id, channels, opt, max_ngen):
    if channels is None:
        nchannels = 'all'
    else:
        nchannels = len(channels)
    cp_filename = Path('checkpoints') / f'random_{sample_id}' / f'{feature_set}_off{opt.offspring_size}_' \
                                                                f'ngen{max_ngen}_{nchannels}chan.pkl'
    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)
    if cp_filename.is_file():
        print(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        print(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False
    t_start = time.time()
    final_pop, halloffame, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_filename, continue_cp=continue_cp)
    t_stop = time.time()
    print('Optimization time', t_stop - t_start)

    output = {'final_pop': final_pop, 'halloffame': halloffame, 'log': log, 'hist': hist}
    return output


# Define map function. To start ipyparallel pool, in a new terminal run: `ipcluster start -n N`
try:
    rc = Client()  # profile=os.getenv('IPYTHON_PROFILE')
    lview = rc.load_balanced_view()

    def mapper(func, it):
        start_time = time.time()
        ret = lview.map_sync(func, it)
        print(f'Generation took {time.time() - start_time} s - {datetime.now()}')
        return ret
except:
    mapper = None

if mapper is not None:
    print(f'IPyparallel started successfully with {len(rc)} parallel processes')

map_function = mapper


# load param files
random_params_file = 'config/params/random.csv'
random_params = pd.read_csv(random_params_file, index_col='index')

for sample_id in sample_ids:
    for feature_set in feature_sets:
        print(f"Starting: sample id {sample_id} - feature set {feature_set} - "
              f"max_ngen {max_ngen} - pop size {offspring_size} --- {datetime.now()}")
        t_start = time.time()
        prep = l5pc_evaluator.prepare_optimization(feature_set, sample_id, offspring_size=offspring_size, channels=channels,
                                                   map_function=map_function)
        opt = prep['optimisation']
        evaluator = prep['evaluator']
        fitness_calculator = prep['objectives_calculator']
        fitness_protocols = prep['protocols']


        out = l5pc_evaluator.run_optimization(feature_set, sample_id, channels, opt, max_ngen=max_ngen)
        final_pop = out['final_pop']
        halloffame = out['halloffame']
        log = out['log']
        hist = out['hist']

        best_params = evaluator.param_dict(halloffame[0])
        original_params = random_params.loc[f'random_{sample_id}'].to_dict()

        # evaluate
        best_responses = evaluator.run_protocols(protocols=fitness_protocols.values(), param_values=best_params)
        original_responses = evaluator.run_protocols(protocols=fitness_protocols.values(), param_values=original_params)

        fig = l5pc_plot.plot_multiple_responses([original_responses, best_responses], return_fig=True)

        if save_fig:
            fig_folder = Path('figures') / f'random_{sample_id}'

            if not fig_folder.is_dir():
                os.makedirs(fig_folder)

            fig.savefig(fig_folder / f'{feature_set}_original_best.pdf')

        t_stop = time.time()
        print(f"Finished: sample id {sample_id} - feature set {feature_set} - max_ngen {max_ngen} "
              f"- pop size {offspring_size} in {t_stop - t_start} --- {datetime.now()}")
