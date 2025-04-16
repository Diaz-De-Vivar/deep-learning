#!/bin/bash

# chmod +x setup_deep_learning_env.sh
# ./setup_deep_learning_env.sh

# Initialize Conda environment activation if not already available
# This command is necessary for the "conda activate" command to work in a script.
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "Creating the deep_learning conda environment..."
conda create -n deep_learning python=3.9 -y

echo "Activating the deep_learning environment..."
conda activate deep_learning

echo "Installing scientific and data libraries..."
conda install -c conda-forge numpy scipy matplotlib scikit-learn pandas jupyterlab seaborn h5py gym -y

echo "Installing TensorFlow and Keras..."
conda install -c conda-forge tensorflow keras -y

echo "Installing PyTorch with CUDA support (pytorch-cuda=11.7) along with torchvision and torchaudio..."
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.7 -y

echo "Environment setup complete."