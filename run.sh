#!/bin/bash
#
#SBATCH -p ALL
#SBATCH -c 1 # number of cores
pip install jax
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cpu.html

python3 data_preperation.py