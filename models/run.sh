#!/bin/bash

#for model in 'hay' 'hallerman'; do

export OPT_MODEL='cultured'

#  for feature_set in 'soma' 'multiple' 'extra'; do

      #export OPT_FEATURE_SET=${feature_set}
      export OPT_FEATURE_SET='soma'

      for seed in {1..10}; do

          export OPT_SEED=${seed}

#          for sample in 0 1 2 3 ; do

              #export OPT_SAMPLE_ID=${sample}
              export OPT_SAMPLE_ID=0

              sbatch ipyparallel.sbatch

#          done
      done
#  done
#done

