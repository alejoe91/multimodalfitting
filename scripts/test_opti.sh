export OPT_MODEL='hay_ais'  # hay | hay_ais | experimental
export OPT_FEATURE_SET='extra'
export OPT_EXTRA_STRATEGY='sections'
export OPT_SEED=1

# Define model and experimental folders here. They should contain:
# features_BPO.json, protocols_BPO.json, probe_BPO.json,
exp_folder='../data/experimental/210301_3113_cell1/efeatures'
hay_folder='../data/models/hay_ecode_probe_planar/efeatures'
hayais_folder='../data/models/hay_ais_ecode_probe_planar/efeatures'

export CELL_FOLDER="../cell_models"
export OPT_FOLDER="../optimization_results"

# set correct folder for optimization
if [ $OPT_MODEL == "experimental" ];
then
  export DATA_FOLDER=$exp_folder
elif [ $OPT_MODEL == "hay" ];
then
  export DATA_FOLDER=$hay_folder
elif [ $OPT_MODEL == "hay_ais" ];
then
  export DATA_FOLDER=$hayais_folder
fi

/gpfs/bbp.cscs.ch/project/proj38/home/damart/LFPy/lfpyenv/bin/python3 run_optimizations.py --cell-folder=${CELL_FOLDER}     \
                                                                --seed=${OPT_SEED}                        \
                                                                --feature-set=${OPT_FEATURE_SET}          \
                                                                --model=${OPT_MODEL}                      \
                                                                --data-folder=${DATA_FOLDER}              \
                                                                --opt-folder=${OPT_FOLDER}                \
                                                                --extra-strategy=${OPT_EXTRA_STRATEGY}
