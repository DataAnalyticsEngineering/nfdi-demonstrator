#####################################
# Docker configuration for the NFDI demonstrator by Data Analytics in Engineering
# 
# Build image:
# $ docker build -t unistuttgartdae/nfdi-demonstrator .
#
# Push image to registry (after docker login):
# $ docker push unistuttgartdae/nfdi-demonstrator
#
# Pull image from registry:
# $ docker pull unistuttgartdae/nfdi-demonstrator
#
# Run a docker container based on the image:
# $ docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace/nfdi-demonstrator unistuttgartdae/nfdi-demonstrator
#
# Alternatively, start a container daemon using docker compose:
# $ docker compose up -d
# Then attach to the running container daemon:
# $ docker compose exec nfdi-demonstrator bash
# Stop the container daemon again:
# $ docker compose down
#####################################

# Set the base image
# This container uses PyTorch 2.5.1 and CUDA 12.4 - Feel free to change to your environment
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Define variables
ENV WORKSPACE_DIR="/workspace"
ENV PROJECT_DIR="${WORKSPACE_DIR}/nfdi-demonstrator"
ENV GIT_REPO="git@github.com:DataAnalyticsEngineering/nfdi-demonstrator.git"

# Install dependencies
RUN apt update && \
    apt install git -y

# Set working directory
WORKDIR ${PROJECT_DIR}

RUN git clone ${GIT_REPO} && \
    pip install -r requirements.txt

ENTRYPOINT ["python3 -m jupyter lab"]
