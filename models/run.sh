#!/bin/bash

#for model in 'hay' 'hallerman'; do

module load unstable
module load py-mpi4py/3.0.3

export OPT_MODEL='experimental'  # hay | hay_ais | experimental

for feature_set in 'soma' 'extra'; do

      export OPT_FEATURE_SET=${feature_set}

      for seed in {1..10}; do

          export OPT_SEED=${seed}

              export OPT_SAMPLE_ID=0

              sbatch ipyparallel.sbatch

      done
done

