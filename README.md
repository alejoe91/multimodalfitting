# Multi-modal fitting of multi-compartment models

This repo explores the use of multimodal data to construct biophisically detailed multicompartment models.
In particular, we the use patch-clamp data and HD-MEA recordings.

## Organization of the repository

The `multimodalfitting` folder is divided in different folders with different purposes:

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


## Scripts

The `scripts` folder contain the executable files to run the optimization jobs.

## Notebooks

The `notebooks` folder contains all analysis notebooks to reproduce the manuscript figure. They should be run in order,
based on the number of the file name. Check out the `notebooks/README.md` for more details.

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
- Aurlien Jaquier

#### Experiments

- Alessio Buccino (ETH)
- Julian Bartram (ETH)
- Xiaohan Xue (ETH)
- Vishalini Emmenegger (ETH)
- Tobias Gänswein

### Cite

For further information please refer to the paper in [Neural Computation]( https://doi.org/10.1162/neco_a_01672)

If you use the software, please cite:

```
@article{buccino2024multimodal,
  title={A Multimodal Fitting Approach to Construct Single-Neuron Models with Patch Clamp and High-Density Microelectrode Arrays},
  author={Buccino, Alessio Paolo and Damart, Tanguy and Bartram, Julian and Mandge, Darshan and Xue, Xiaohan and Zbili, Mickael and G{\"a}nswein, Tobias and Jaquier, Aur{\'e}lien and Emmenegger, Vishalini and Markram, Henry and others},
  journal={Neural Computation},
  pages={1--46},
  year={2024},
  publisher={MIT Press 255 Main Street, 9th Floor, Cambridge, Massachusetts 02142, USA~…}
}
```

### Ackowledgments

The study associated to this repo was supported by the ETH Zurich Postdoctoral Fellowship 19-2 FEL-17 (APB), the ERC Advanced Grant 694829 "neuroXscales" (JB, XX, TG, VE, AH), the China Scholarship Council (XX), by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology (TD, DM, MZ, AJ, HM, WVG), and by the European Union’s Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3) (TD, AJ).


Copyright (c) 2022-2024 Blue Brain Project/EPFL