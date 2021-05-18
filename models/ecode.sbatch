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
source /gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/activate

LOGS=./logs/${SLURM_JOBID}
mkdir -p $LOGS

export IPYTHONDIR=./.ipython
mkdir -p ${IPYTHONDIR}

export IPYTHON_PROFILE=benchmark.${SLURM_JOBID}

/gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/ipcontroller --init --ip='*' --profile=${IPYTHON_PROFILE} &
sleep 20
srun --output="${LOGS}/engine.out" /gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/ipengine --profile=${IPYTHON_PROFILE} &
sleep 20

/gpfs/bbp.cscs.ch/home/mandge/lfpenv/bin/python3 run_ecode.py


run_optimizations.py --ipyparallel                    \
                                                                         --seed=${OPT_SEED}               \
                                                                         --feature_set=${OPT_FEATURE_SET} \
                                                                         --sample_id=${OPT_SAMPLE_ID}     \
                                                                         --model=${OPT_MODEL}
1-generate-ecode-responses-and-features-Copy1.py