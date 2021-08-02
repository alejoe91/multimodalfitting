#!/bin/bash

#SBATCH --constraint=cpu
#SBATCH --ntasks=20
#SBATCH --partition=prod
#SBATCH --job-name=expt_soma_2
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

/gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun --output="${LOGS}/engine.out" /gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/ipengine --profile=${IPYTHON_PROFILE} &
sleep 20

/gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/python3 run_optimizations.py --ipyparallel                    \
                                                                      --seed=${OPT_SEED}               \
                                                                      --feature_set=${OPT_FEATURE_SET} \
                                                                      --model=${OPT_MODEL}             \
                                                                      --folder=${OPT_FOLDER}
