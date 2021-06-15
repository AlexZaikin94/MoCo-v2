#!/bin/bash

# Setup env
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hw1
echo "start evaluation"
python evaluate.py
echo "done evaluation"