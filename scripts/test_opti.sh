export OPT_MODEL='hay'
export OPT_FEATURE_SET='extra'
export OPT_EXTRA_STRATEGY='all'
export OPT_SEED=1

export CELL_FOLDER="../cell_models"
export OPT_FOLDER="../optimization_results"

# if ABD is true, the axon_bearing_dendrite is added to the sections
export OPT_ABD=0
# if RA is true, the Ra for AIS and ABD are optimized separately
export OPT_RA=0

/gpfs/bbp.cscs.ch/project/proj38/home/damart/LFPy_NEW/lfpyenv_new/bin/python3 run_optimizations.py \
                                                    --cell-folder=${CELL_FOLDER}      \
                                                    --seed=${OPT_SEED}                \
                                                    --feature-set=${OPT_FEATURE_SET}  \
                                                    --model=${OPT_MODEL}              \
                                                    --opt-folder=${OPT_FOLDER}        \
                                                    --abd=${OPT_ABD}                  \
                                                    --ra=${OPT_RA}                    \
                                                    --extra-strategy=${OPT_EXTRA_STRATEGY}
