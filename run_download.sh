#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hw1
echo "start download_data"
python download_data.py
echo "done download_data"