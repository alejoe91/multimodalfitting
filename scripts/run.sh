export OPT_MODEL='hay'  # hay | hay_ais | experimental

export SIM='lfpy' # lfpy | neuron
# Define model and experimental folders here. They should contain:
# features_BPO.json, protocols_BPO.json, probe_BPO.json,
exp_folder='../data/experimental/210301_3113_cell1/efeatures'
hay_folder='../data/models/hay_ecode_probe_planar/efeatures'
hayais_folder='../data/models/hay_ais_ecode_probe_planar/efeatures'

export CELL_FOLDER="../cell_models"
export OPT_FOLDER="../optimization_results"

# if ABD is true, the axon_bearing_dendrite is added to the sections
export ABD=false
# if RA is true, the Ra for AIS and ABD are optimized separately
export RA=false

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

export OPT_FEATURE_SET='soma'
export OPT_EXTRA_STRATEGY='all'
for seed in {1..10}; do
  export OPT_SEED=${seed}
  sbatch ipyparallel.sbatch
done

export OPT_FEATURE_SET='extra'
for strategy in 'all' 'single' 'sections'; do
      export OPT_EXTRA_STRATEGY=${strategy}
      for seed in {1..10}; do
          export OPT_SEED=${seed}
          sbatch ipyparallel.sbatch
      done
done