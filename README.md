# Mesh Neural Cellular Automata (SIGGRAPH 2024) [[Project Page](https://meshnca.github.io/)]

[![arXiv](https://img.shields.io/badge/arXiv-2108.00946-b31b1b.svg)](https://arxiv.org/abs/2311.02820)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IVRL/MeshNCA/blob/main/notebooks/colab.ipynb)

This is the official implementation of the paper titled "Mesh Neural Cellular Automata".

![summary of results](data/teaser.gif)

## Getting Started

For a quick and hands-on experimentation with MeshNCA, we suggest trying our Google Colab notebook (click on Open in
Colab button at the top of the readme).
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

## How to run

Here, we outline the steps to train and visualize the MeshNCA model.

To run the code you will need to have a yaml config file that specifies the training settings.
You can find example config files in the `configs` directory.

For example, to train the MeshNCA model with PBR textures config, you can use the following command:

```bash
python3 train_image.py --config configs/pbr_texture.yaml
```

### Config files

Each setting in the config file is explained in the comments.
Currently, we only support the image-guided training scheme and provide two sample config files for training with PBR
textures and single RGB textures.
The configuration and the training code for text-guided and motion-guided training schemes will be released soon.

* `configs/pbr_texture.yaml`: Contains the training settings for PBR textures.
  You can use this config file when you want MeshNCA to simultaneously learn and synthesize multiple related textures
  such as the albedo, normal, roughness, height, and ambient occlusion maps.
* `configs/single_texture.yaml`: Contains the training settings for RGB textures.

#### Example:

The target images are specified in the `appearance_loss_kwargs` section of the config file.
The `target_images_path` dictionary contains the paths to the target images,
and the `num_channels` dictionary specifies the number of channels in each target image.

```yaml
loss:
  appearance_loss_kwargs:
    # The single channel images will be expanded to 3 channels for evaluating the VGG-based style loss
    target_images_path: {
      "albedo": "data/pbr_textures/Abstract_008/albedo.jpg",
      "height": "data/pbr_textures/Abstract_008/height.jpg",
      "normal": "data/pbr_textures/Abstract_008/normal.jpg",
      "roughness": "data/pbr_textures/Abstract_008/roughness.jpg",
      "ambient_occlusion": "data/pbr_textures/Abstract_008/ambient_occlusion.jpg",
    }
    # Number of channels in the target images
    num_channels: {
      "albedo": 3,
      "height": 1,
      "normal": 3,
      "roughness": 1,
      "ambient_occlusion": 1,
    }
```

If you face out of memory issues, you can reduce the `num_views` or the `batch_size` in the config file.

```yaml
train:
  batch_size: 1
  camera:
    num_views: 6  
```

To record the training logs in wandb (Weights and Biases), you can set the `wandb` key in the config file.
```yaml
wandb:
  project: "Project Name"
  key: "Your API Key"
```

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
