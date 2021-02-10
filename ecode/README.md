# Hay model optimization

This folder contains python scripts and notebooks to investigate different optimization strategies for the 
[Hay model](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002107) of a L5PC cell.  

The notebooks / scripts need to be run in the following order:

1. 1-random-sampling-and-features.ipynb  -  Perform random sampling of the release parameter space 
   to create `test_models` used for evaluation and compute features
2. 2-run_optimizations.py  -  Run optimizations for the different feature sets and test models
3. 3-analyze_optimizations_create_runs.pkl.ipynb  -  Parse optimization output to be further analyzed 
4. 4-analyze-optimization-performance.ipynb  -  Analyze performance of the optimizations

Further information can be found in the different notebooks.

