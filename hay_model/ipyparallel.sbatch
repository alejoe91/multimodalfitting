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
export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

/gpfs/bbp.cscs.ch/home/damart/venv_lfpy/bin/ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun --output="${LOGS}/engine.out" /gpfs/bbp.cscs.ch/home/damart/venv_lfpy/bin/ipengine --profile=${IPYTHON_PROFILE} &
sleep 20

touch ${LOGS}/params.dat
echo ${FEATURE_SET} >> ${LOGS}/params.dat
echo ${SAMPLE_ID} >> ${LOGS}/params.dat
echo ${OPT_SEED} >> ${LOGS}/params.dat

/gpfs/bbp.cscs.ch/home/damart/venv_lfpy/bin/python3 opt.py -v --offspring_size=250 --seed=${OPT_SEED} --ipyparallel --feature_set=${FEATURE_SET} --sample_id=${SAMPLE_ID}

