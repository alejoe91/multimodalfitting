#!/bin/bash
# OLD: source /gpfs/bbp.cscs.ch/project/proj38/LFP_project/lfpyenv_new/bin/

#New 25 June 2023: created using /gpfs/bbp.cscs.ch/project/proj38/LFP_project/create_venv.sh
source /gpfs/bbp.cscs.ch/project/proj38/LFP_project/june23_lfpy/bin/activate

export OPT_MODEL='cell1_211011_3436'  # hay | hay_ais | experimental | cell1_210301_3113 | cell1_211006_3148 | cell1_211011_3436

export SIM='lfpy' # lfpy | neuron

export OPT_FOLDER="../optimization_results"
export CELL_FOLDER="../cell_models"

# if ABD is true, the axon_bearing_dendrite is added to the sections
export OPT_ABD=1
# if RA is true, the Ra for AIS and ABD are optimized separately
export OPT_RA=0
#if CM_RA then, cm is optimised for all sections seperately and Ra as global
export OPT_CM_RA=0

for strategy in 'all' 'single' 'sections'; do
      export OPT_STRATEGY=${strategy}
      for seed in {1..10}; do
          export OPT_SEED=${seed}
          sbatch ipyparallel.sbatch
      done
done
