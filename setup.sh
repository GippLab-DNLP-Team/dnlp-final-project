#!/bin/bash -i
set -e

# Set up Conda, install Python
conda create -n dnlp python=3.10
conda activate dnlp

# Check for CUDA and install appropriate PyTorch version
if command -v nvidia-smi &>/dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support."
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
else
    echo "CUDA not detected, installing CPU-only PyTorch."
    conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
fi

# Install additional packages
conda install "numpy<2" "mkl<2025" tqdm==4.66.2 requests==2.31.0 transformers==4.38.2 tensorboard==2.16.2 tokenizers==0.15.1 -c conda-forge -c huggingface
pip install explainaboard-client==0.1.4 sacrebleu==2.4.0 scikit_learn==1.6.1
