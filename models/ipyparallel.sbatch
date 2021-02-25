#!/bin/bash

#SBATCH --constraint=cpu
#SBATCH --ntasks=20
#SBATCH --partition=prod
#SBATCH --job-name=LFPy
#SBATCH --time=0-24:00:00
#SBATCH --error=./logs/%A.stdout
#SBATCH --output=./logs/%A.stdout
#SBATCH --account=proj38

set -e
set -x

LOGS=./logs/${SLURM_JOBID}
mkdir -p $LOGS

export IPYTHONDIR=./.ipython
mkdir -p IPYTHONDIR

export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

/gpfs/bbp.cscs.ch/home/damart/venv_lfpy/bin/ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun --output="${LOGS}/engine.out" /gpfs/bbp.cscs.ch/home/damart/venv_lfpy/bin/ipengine --profile=${IPYTHON_PROFILE} &
sleep 20

/gpfs/bbp.cscs.ch/home/damart/venv_lfpy/bin/python3 2-run-optimizations.py --ipyparallel                    \
                                                                           --seed=${OPT_SEED}               \
                                                                           --feature_set=${OPT_FEATURE_SET} \
                                                                           --sample_id=${OPT_SAMPLE_ID}     \
                                                                           --model=${OPT_MODEL}
