#!/bin/bash

source /miniconda/etc/profile.d/conda.sh
conda activate GSTK

# Run any pip commands
# These can only be ran on the loaded image when a GPU is present.
pip install src/submodules/CUDA_renderer
pip install src/submodules/CUDA_knn
pip install pygfx
pip install glfw

nvidia-smi

# Then, execute the Docker CMD
exec "$@"