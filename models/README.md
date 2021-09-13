# Hay model optimization

This folder contains python scripts and notebooks to investigate different optimization strategies for the 
[Hay model](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002107) of a L5PC cell and 
modifications of this model that contain an axon initial segment (HAY_AIS) or an axon initial segment + an aoxn hillock 
(HAY_AIS_HILLOCK).
In addition, it also supports optimization of experimental data.

## Usage

The notebooks / scripts need to be run in the following order:

- `0-test-*.ipynb` Test different models on the ECODE protocols
- `1-generate-ecode-responses-*.ipynb` - Generate responses, intra- and extracellular features for optimization
- `run_optimizations.py`  -  Run optimizations for the different feature sets and test models
- `2-create_runs_pkl_from_optimizations.ipynb`  -  Parse optimization output to pkl files for analysis 
- `3-analyze-optimization-*.ipynb`  -  Analyze performance of the optimizations

Further information can be found in the different notebooks.

