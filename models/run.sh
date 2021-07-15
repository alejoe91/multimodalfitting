#!/bin/bash

export OPT_MODEL='hay_ais'

for feature_set in 'soma' 'extra'; do
    export OPT_FEATURE_SET=${feature_set}
    for seed in {1..10}; do
        export OPT_SEED=${seed}
        sbatch ipyparallel.sbatch
    done
done
