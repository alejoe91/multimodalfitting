#!/bin/bash

#SBATCH --constraint=cpu
#SBATCH --ntasks=20
#SBATCH --partition=prod
#SBATCH --job-name=LFPy
#SBATCH --time=24:00:00
#SBATCH --error=./logs/%A.stdout
#SBATCH --output=./logs/%A.stdout
#SBATCH --account=proj38

set -e
set -x

LOGS=./logs/${SLURM_JOBID}
mkdir -p $LOGS

export IPYTHONDIR=./.ipython
mkdir -p ${IPYTHONDIR}

export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

/gpfs/bbp.cscs.ch/project/proj38/home/damart/LFPy_NEW/lfpyenv_new/bin/ipcontroller --init --ip='*' --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 30
srun --mpi=none /gpfs/bbp.cscs.ch/project/proj38/home/damart/LFPy_NEW/lfpyenv_new/bin/ipengine --timeout=300 --profile=${IPYTHON_PROFILE} &
sleep 30

/gpfs/bbp.cscs.ch/project/proj38/home/damart/LFPy_NEW/lfpyenv_new/bin/python3 run_optimizations.py \
                             --ipyparallel                            \
                             --cell-folder=${CELL_FOLDER}             \
                             --seed=${OPT_SEED}                       \
                             --feature-set=${OPT_FEATURE_SET}         \
                             --model=${OPT_MODEL}                     \
                             --opt-folder=${OPT_FOLDER}               \
                             --extra-strategy=${OPT_EXTRA_STRATEGY}   \
                             --sim=${SIM}                             \
                             --abd=${OPT_ABD}                         \
                             --ra=${OPT_RA}

