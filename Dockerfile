# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git wget  \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda to manage Python and environments
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh

# Add Conda to PATH
ENV PATH="/miniconda/bin:${PATH}"

# Install the repo
RUN git clone https://github.com/skywolf829/GSTK_backend.git /app

# Move working directory (cd)
WORKDIR /app

# Create a Conda environment
RUN conda env create -f env.yml

# Activate the environment and install any additional packages with pip or conda as needed
RUN echo "source activate GSTK_backend" > ~/.bashrc
ENV PATH /opt/conda/envs/GSTK_backend/bin:$PATH


# Command to run your application
CMD ["python", "src/backend_web.py"]
