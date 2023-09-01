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


## Notebooks

The `notebooks` folder contains all analysis notebooks to reproduce the manuscript figure.

## Requirements

- [neuron](https://www.neuron.yale.edu/neuron/): widely used simulator for multicompartment models
- [LFPy](https://lfpy.readthedocs.io/en/latest/): Python package to compute extracellular signals from NEURON 
simulations
- [BluePyOpt](https://github.com/BlueBrain/BluePyOpt): Blue Brain Optimization Library to perform optimizations
- [BluePyEfe](https://github.com/BlueBrain/BluePyEfe): Blue Brain library for feature extraction 
- [neuroplotlib](https://github.com/LFPy/neuroplotlib): plotting library built on top of LFPy


## Authors

This project is a collaboration between the BEL Lab @ ETH-BSSE  and the Blue Brain Project @ EPFL:

#### Modeling

- Alessio Buccino (ETH)
- Tanguy Damart (BBP)
- Darshan Mandge (BBP)
- Mickael Zbili (BBP)
- Werner Van Geit (BBP)

#### Experiments

- Alessio Buccino (ETH)
- Julian Bartram (ETH)
- Xiaohan Xue (ETH)

### Cite

For further information please refer to the preprint on [bioRxiv]( https://doi.org/10.1101/2022.08.03.502468)

If you use the software, please cite:
```
@article{buccino2022multi,
  title={A multi-modal fitting approach to construct single-neuron models with patch clamp and high-density microelectrode arrays},
  author={Buccino, Alessio Paolo and Damart, Tanguy and Bartram, Julian and Mandge, Darshan and Xue, Xiaohan and Zbili, Mickael and G{\"a}nswein, Tobias and Jaquier, Aur{\'e}lien and Emmenegger, Vishalini and Markram, Henry and others},
  journal={bioRxiv},
  pages={2022--08},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

