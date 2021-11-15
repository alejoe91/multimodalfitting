#!/bin/bash

#SBATCH --constraint=cpu
#SBATCH --ntasks=30
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
mkdir -p ${IPYTHONDIR}

export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun ipengine --profile=${IPYTHON_PROFILE} &
sleep 20

python3 run_ecode.py
