#!/bin/bash

for model in 'hay' 'hallerman'; do

  export OPT_MODEL=${model}

  for feature_set in 'soma' 'multiple' 'extra'; do

      export OPT_FEATURE_SET=${feature_set}

      for seed in {1..5}; do

          export OPT_SEED=${seed}

          for sample in 0 1 2 3 ; do

              export OPT_SAMPLE_ID=${sample}

              sbatch ipyparallel.sbatch

          done

      done

  done

done

