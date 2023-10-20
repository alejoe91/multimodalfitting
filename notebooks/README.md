# Organization of notebooks

The notebooks in this folder are used to analyze the data after the optimization.

Notebooks starting qith `0-` perform general tasks, such as testing a model, correcting a morphology, or exporting to NWB.

The notebooks starting from `1-` should be run in sequence. 

- `1-generate-*.ipynb`: the first notebook uses the model or the experimental data to generate feature and protocols file. 
  | Note that the different models (ground-truth, cell1, cell2) have different notebooks. Running this notebooks is required 
  | before running the optimization scripts.

- `2-create-runs-pkl-from-checkpoints.ipynb`: this notebook can be run after the optimization scripts have run. It converts 
  | the optimization checkpoint files to a well-organized pickle file that is used for later analysis.

- `3-compute-all-responses.ipynb`: this notebook computes all the responses from the optmized solutions.

- `4-optimization-performance.ipynb`: this notebooks assess the performance of the optimization for all seeds (on the "training" 
  | protocols) and for the best seed on the validation protocols. It generates all panels of Figures 4-5 (`ground-truth` folder), 
  | and Figures 7 and 8 (`experimental/cell1` and `experimental/cell2`, respectively).


The remaining notebooks are additional supplementary analysis.