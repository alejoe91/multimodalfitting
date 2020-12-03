#!/bin/bash

for feature_set in 'soma' 'multiple' 'extra'; do
    export FEATURE_SET=${feature_set}
    for seed in {1..5}; do
        export OPT_SEED=${seed}
        for sample in 0 1 2 3 ; do
            export SAMPLE_ID=${sample}
            sbatch ipyparallel.sbatch
        done
    done
done
