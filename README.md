# Multi-modal fitting of multi-compartment models

This repo explores the use of multimodal data to construct biophisically detailed multicompartment models.
In particular, we the use patch-clamp data and HD-MEA recordings.

## Organization of the repository

The repo is divided in different folders with different purposes:

### models

This is the core of the repo. It contains scrips and notebooks to run optimizations using BluePyOpt on several test 
models and on experimental data. See the README in the folder for more details. 

### ecode

This module contains the definition of the E-Code protocols developed at the BBP for stimulating the cells (real
and virtual) for optimization.

### efeatures_extraction

This module contains the tools to extract intracellular and extracellular features from the data in preparation 
for the optimization.

### imaging_tools

This module contains scripts to clean up experimentally-obtained cell reconstructions.

## Requirements

- [neuron](https://www.neuron.yale.edu/neuron/): widely used simulator for multicompartment models
- [LFPy](https://lfpy.readthedocs.io/en/latest/): Python package to compute extracellular signals from NEURON 
simulations
- [BluePyOpt](https://github.com/BlueBrain/BluePyOpt): Blue Brain Optimization Library to perform optimizations
- [BluePyEfe](https://github.com/BlueBrain/BluePyEfe): Blue Brain library for feature extraction 
- [neuroplotlib](https://github.com/LFPy/neuroplotlib): plotting library built on top of LFPy

## Installation

If you want to use the present package, you first need to install NEURON with Python support on your machine.

And then the package itself:

```
    git clone https://github.com/alejoe91/multimodalfitting
    pip install -e multimodalfitting
```

## Funding & Acknowledgement

