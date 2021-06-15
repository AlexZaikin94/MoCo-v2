#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hw1
echo "start moco job"
nvidia-smi
python main_moco.py
echo "done job"