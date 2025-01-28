# NFDI Demonstrator

This repository contains a demonstrator for NFDI-Matwerk by the Heisenberg Professorship Data Analytics in Engineering at the University of Stuttgart.

While it is assumed in many applications that components are characterized by a homogeneous microstructure, this is not always the case.
In fact, materials often exhibit heterogeneities, which can affect the material behavior drastically.
In computational homogenization, the overall goal is to use numerical simulations to determine the effective material behavior of a heterogeneous material based on a given microstructure.

This demonstrator showcases a thermal homogenization problem of a 2D microstructure.
An interactive widget allows the user to play around with different parameters of the homogenization problem and solves it in near real-time (<10ms on state-of-the-art GPUs) to observe their implications.
Behind the scenes, a high-fidelity simulation using the Finite Element Method (FEM) on a 400x400 grid (given directly by the microstructure image) is carried out with a GPU-accelerated implementation of our *FANS-CG* solver that features a special FFT-based preconditioner tailored to this problem.

![Interactive widget](data/widget-screenshot.png?raw=true "Interactive widget")

Jupyter notebook with the interactive widget and additional examples: [demonstrator.ipynb](demonstrator.ipynb)

## How to get started

While this demonstrator is meant to run on a GPU (ideally recent NVIDIA architecture), it can also run on the CPU (but slower).

### Manual installation

1. Install dependencies

- This demonstrator requires a Python installation with version >=3.11. The Python version can be changed, e.g., using [`pyenv`](https://github.com/pyenv/pyenv).
- Since this demonstrator is based on PyTorch, the prerequisites on <https://pytorch.org/get-started/locally/> apply.
- Optional: To compile the PyTorch model, a C++ compiler is required. On Ubuntu, the `build-essential` package is enough.

2. Clone repository

```
git clone https://github.com/DataAnalyticsEngineering/nfdi-demonstrator.git
cd nfdi-demonstrator
```

3. Create and activate virtual environment (recommended)

```
python3 -m venv venv
. ./venv/bin/activate
```

4. Install dependencies in virtual environment:

```
pip install -r requirements.txt
```

5. Start Jupyter Lab server:

```
jupyter lab
```

### Docker container

We provide a Docker image based on NVIDIA's PyTorch image that runs our demonstrator out of the box:

```
docker run -it --gpus all --ipc=host --net=host unistuttgartdae/nfdi-demonstrator
```

Or clone the repository and start the container automatically using `docker compose`:

```
git clone https://github.com/DataAnalyticsEngineering/nfdi-demonstrator.git
cd nfdi-demonstrator
docker compose up
```

Note that the underlying the base image by NVIDIA is rather large in size.

## Acknowledgements

**Authors**

- Julius Herb <herb@mib.uni-stuttgart.de>
- Sanath Keshav <keshav@mib.uni-stuttgart.de>
- Felix Fritzen <fritzen@mib.uni-stuttgart.de>

**Affiliation**

> Heisenberg Professorship Data Analytics in Engineering
> 
> Institute of Applied Mechanics
> 
> University of Stuttgart
> 
> Universitätsstr. 32, 70569 Stuttgart
> 
> https://www.mib.uni-stuttgart.de/dae

**Funding acknowledgments**

>- Contributions by Felix Fritzen are partially funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy - EXC 2075 – 390740016. Felix Fritzen is funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) within the Heisenberg program DFG-FR2702/8 - 406068690 and DFG-FR2702/10 - 517847245.
>
>- Contributions of Julius Herb are partially funded by the Ministry of Science, Research and the Arts (MWK) Baden-Württemberg, Germany, within the Artificial Intelligence Software Academy (AISA).
>
>- This work is also supported by / part of the consortium NFDI-MatWerk, funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under the National Research Data Infrastructure - NFDI 38/1 - project number 460247524
>
>- The authors acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).
