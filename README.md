# Mesh Neural Cellular Automata (SIGGRAPH 2024) [[Project Page](https://meshnca.github.io/)]

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2311.02820)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IVRL/MeshNCA/blob/main/notebooks/colab.ipynb) 


This is the official implementation of the paper titled "Mesh Neural Cellular Automata".

![summary of results](data/teaser.gif)

## Getting Started

For a quick and hands-on experimentation with MeshNCA, we suggest trying our Google Colab notebook. 
You can follow the steps below to run our code locally for in-depth experimentation. 

### Installing the packages

1. Clone the repository

```bash
git clone https://github.com/IVRL/MeshNCA.git
cd MeshNCA
```

2. Create a new virtual environment and install the required python packages.
    * We Use Kaolin for differentiable rendering. Currently, Kaolin only supports PyTorch >= 1.8, <= 2.1.1
    * We use PyG (PyTorch Geometric) to implement the message passing in MeshNCA architecture.

```bash
python3 -m venv env
source env/bin/activate
chmod +x install_requirements.sh
./install_requirements.sh
```

### Downloading the data

Run the `download_data.py` script to download the data.

```bash
python3 download_data.py
```

The data will be downloaded to the `data` directory and has three parts:

* The mesh dataset consisting of 48 meshes with valence close to 6 and uniformly distributed vertices.
* 72 Physically-based-rendering (PBR) textures, each containing 5 textures (albedo, normal, roughness, height, and
  ambient occlusion) All textures have a resolution of 1024x1024.
* 45 RGB textures adopted from [DyNCA](https://dynca.github.io/).


### Running the code


## License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
