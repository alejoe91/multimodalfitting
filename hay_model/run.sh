#!/bin/bash

export FEATURE_SET='multiple' # multiple, soma, extra

for seed in {1..5}; do
    export OPT_SEED=${seed}
    for sample in 0 1 2 3 ; do
        export SAMPLE_ID=${sample}
        sbatch ipyparallel.sbatch
    done
done
