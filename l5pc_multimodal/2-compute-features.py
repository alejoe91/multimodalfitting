# 2) Compute features for random parameters
#
# Using the generated random parameters, we next compute responses and features for 3 different feature sets:
#
# - 'bap': features extracted from somatic recording with 3 different steps + a pulse response measured at the soma plus
#    two locations on the apical dendrite (Backpropagating Action Potential)
#
# - 'soma': features extracted from somatic recording with 3 different steps
#
# - 'extra': features extracted from somatic recording with 3 different steps and from mean extracellular
#    action potential
#
# Computed features are saved in the `config/features/` folder for each parameter set and are ready to be used
# for optimization.

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
import pandas as pd
from pathlib import Path

matplotlib.use('agg')

import l5pc_model
import l5pc_evaluator
import l5pc_plot

# Define extracellular electrodes
#
# Importantly, if the electrode design changes, new features need to be generated.
# However, a subset of channels can be selected before running the optimization procedure.

mea_positions = np.zeros((20, 3))
mea_positions[:, 2] = 20
mea_positions[:, 1] = np.linspace(-500, 1000, 20)
probe = mu.return_mea(info={'pos': list([list(p) for p in mea_positions]), 'center': False, 'plane': 'xy'})
electrode = LFPy.RecExtElectrode(probe=probe, method='linesource')

# Compute features
#
# Features for the different feature sets ('bap', 'soma', 'extra') are listed in the `config/feature_list.json` file.
#
# Here we loop through the different feature sets and random parameters, compute the corresponding features and save
# them in a `json` file that will be later used to construct the `CellEvaluator`.

random_params_file = 'config/params/random.csv'
random_params = pd.read_csv(random_params_file, index_col='index')

feature_sets = ["soma", "bap", "extra"]  # 'soma'/'bap'
channels = None

morphology = ephys.morphologies.NrnFileMorphology('morphology/C060114A7.asc', do_replace_axon=True)
param_configs = json.load(open('config/parameters.json'))
parameters = l5pc_model.define_parameters()
mechanisms = l5pc_model.define_mechanisms()

l5pc_cell = ephys.models.LFPyCellModel('l5pc',
                                       v_init=-65.,
                                       morph=morphology,
                                       mechs=mechanisms,
                                       params=parameters)

param_names = [param.name for param in l5pc_cell.params.values() if not param.frozen]

for feature_set in feature_sets:
    print(f'Feature set {feature_set}')
    responses = []

    if feature_set == "extra":
        fitness_protocols = l5pc_evaluator.define_protocols(electrode)
    else:
        fitness_protocols = l5pc_evaluator.define_protocols()

    if feature_set == "extra":
        sim = ephys.simulators.LFPySimulator(LFPyCellModel=l5pc_cell, cvode_active=True, electrode=electrode)
    else:
        sim = ephys.simulators.LFPySimulator(LFPyCellModel=l5pc_cell, cvode_active=True)

    for i, (index, params) in enumerate(random_params.iterrows()):
        print(f'{i + 1} / {len(random_params)}, {index}')

        feature_folder = f'config/features/{index}'

        feature_file, response = l5pc_evaluator.compute_feature_values(params, l5pc_cell, fitness_protocols, sim,
                                                                       feature_set=feature_set, probe=probe,
                                                                       channels=channels,
                                                                       feature_folder=feature_folder)
        # save probe yaml in the feature folder
        with (Path(feature_folder) / 'probe.json').open('w') as f:
            json.dump(probe.info, f, indent=4)

        responses.append(response)

l5pc_plot.plot_multiple_responses(responses)
