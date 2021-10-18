# Install multimodal fitting in a conda environment with Python3.8

# BluePyOpt fork
pip install https://github.com/alejoe91/BluePyOpt/archive/master.zip

# BluePyEfe BPE2 branch
pip install https://github.com/BlueBrain/BluePyEfe/archive/BPE2.zip

# mpi/mpi4py
conda install -y mpi
conda install -y mpi4py

# neuron 7.8
pip install neuron>=8

# LFPy
pip install LFPy

# other requirements
pip install -r requirements.txt

# multimodalfitting
pip install .
