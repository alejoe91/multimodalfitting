export OPT_MODEL='hay'  # hay | hay_ais | experimental
export OPT_EXTRA_STRATEGY='all'  # all | single | sections

# Define model and experimental folders here. They should contain:
# features_BPO.json, protocols_BPO.json, probe_BPO.json,
exp_folder='../data/experimental/cell1_210301/efeatures'
hay_folder='../data/hay_ecode_probe_planar/efeatures'
hayais_folder='../data/hay_ais_ecode_probe_planar/efeatures'
hayaishillock_folder='../data/hay_ais_hillock_ecode_probe_planar/efeatures'

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
elif [ $OPT_MODEL == "hay_ais_hillock" ];
then
  export DATA_FOLDER=$hayaishillock_folder
fi

for feature_set in 'soma' 'extra'; do
      export OPT_FEATURE_SET=${feature_set}
      for seed in {1..10}; do
          export OPT_SEED=${seed}
          sbatch ipyparallel.sbatch
      done
done
