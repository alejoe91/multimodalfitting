#!/bin/bash

#SBATCH --constraint=cpu
#SBATCH --ntasks=40
#SBATCH --partition=prod
#SBATCH --job-name=LF_MMF
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

ipcontroller --init --ip='*' --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 30
srun --mpi=none ipengine --timeout=300 --profile=${IPYTHON_PROFILE} &
sleep 30
echo "IPython cluster started"

python3 run_optimizations.py \
                             --ipyparallel                            \
                             --cell-folder=${CELL_FOLDER}             \
                             --seed=${OPT_SEED}                       \
                             --strategy=${OPT_STRATEGY}               \
                             --model=${OPT_MODEL}                     \
                             --opt-folder=${OPT_FOLDER}               \
                             --sim=${SIM}                             \
                             --abd=${OPT_ABD}                         \
                             --ra=${OPT_RA}                           \
                             --cm_ra=${OPT_CM_RA}                    
