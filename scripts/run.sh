export OPT_MODEL='hay'  # hay | hay_ais | experimental
export OPT_EXTRA_STRATEGY='all'

# Define model and experimental folders here. They should contain:
# features_BPO.json, protocols_BPO.json, probe_BPO.json,
exp_folder='../data/experimental/cell1_210301/efeatures'
hay_folder='../data/hay_ecode_probe_planar/efeatures'
hayais_folder='../data/hay_ais_ecode_probe_planar/efeatures'

# set correct folder for optimization
if [ $OPT_MODEL == "experimental" ];
then
  export OPT_FOLDER=$exp_folder
elif [ $OPT_MODEL == "hay" ];
then
  export OPT_FOLDER=$hay_folder
elif [ $OPT_MODEL == "hay_ais" ];
then
  export OPT_FOLDER=$hayais_folder
fi

for feature_set in 'soma' 'extra'; do
      export OPT_FEATURE_SET=${feature_set}
      for seed in {1..10}; do
          export OPT_SEED=${seed}
          sbatch ipyparallel.sbatch
      done
done
