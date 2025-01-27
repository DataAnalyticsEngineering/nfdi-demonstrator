# Docker configuration for the NFDI demonstrator by Data Analytics in Engineering
# 
# For end users:
# ==============
# Pull image from registry and run container:
# $ docker run -it --gpus all --ipc=host --net=host unistuttgartdae/nfdi-demonstrator
#
# Alternatively, start the container using docker compose:
# $ docker compose up
#
# Terminate the container using Ctrl+C
#
# For developers:
# ===============
# Build image:
# $ docker build -t unistuttgartdae/nfdi-demonstrator .
#
# Push image to registry (after docker login):
# $ docker push unistuttgartdae/nfdi-demonstrator
#####################################

# Set the base image
# This container uses PyTorch 2.5.1 and CUDA 12.4 - Feel free to change to your environment
FROM nvcr.io/nvidia/pytorch:24.12-py3
#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Define variables
ENV WORKSPACE_DIR="/workspace"
ENV PROJECT_DIR="${WORKSPACE_DIR}/nfdi-demonstrator"
ENV GIT_REPO="https://github.com/DataAnalyticsEngineering/nfdi-demonstrator.git"

# Install dependencies
RUN apt update && \
    apt install -y build-essential git

# Clone demonstrator from GitHub and install dependencies
RUN cd ${WORKSPACE_DIR} && \
    git clone ${GIT_REPO} ${PROJECT_DIR} && \
    cd ${PROJECT_DIR} && \
    pip install -r requirements.txt

# Set working directory
WORKDIR ${PROJECT_DIR}

# Start Juypter Lab server
CMD ["/usr/local/bin/jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
