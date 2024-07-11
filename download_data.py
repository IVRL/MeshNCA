"""
This script downloads the PBR textures and Mesh dataset
"""

import os
import platform
import subprocess

import gdown
import zipfile


def main():
    # Download the datasets
    if not os.path.exists("data/pbr_textures"):
        image_dataset_url = (
            "https://drive.google.com/uc?id=1TsA3fyr1OU_1C4Rk5rtBp-Zzr_54Bofn"
        )
        gdown.download(image_dataset_url, "data/pbr_textures.zip", quiet=False)
        print("Unzipping the PBR Texture dataset")
        with zipfile.ZipFile("data/pbr_textures.zip", "r") as zip_ref:
            zip_ref.extractall("data/pbr_textures")
        os.remove("data/pbr_textures.zip")

    else:
        print("PBR Texture dataset is already downloaded.")
        print("Remove the existing folder at data/pbr_textures to download again.\n")

    if not os.path.exists("data/meshes"):
        mesh_dataset_url = (
            "https://drive.google.com/uc?id=136uliL3tcQinNXg3LXcJB4_Qf-HXg2A3"
        )
        gdown.download(mesh_dataset_url, "data/meshes.zip", quiet=False)
        print("Unzipping the Mesh dataset")
        with zipfile.ZipFile("data/meshes.zip", "r") as zip_ref:
            zip_ref.extractall("data/meshes")
        os.remove("data/meshes.zip")
    else:
        print("Mesh dataset is already downloaded.")
        print("Remove the existing folder at data/meshes to download again.\n")

    if not os.path.exists("data/textures"):
        texture_dataset_url = (
            "https://drive.google.com/uc?id=1HKeRYWAleA4cxQNen90FzBmcDLCxGmZr"
        )
        gdown.download(texture_dataset_url, "data/textures.zip", quiet=False)
        print("Unzipping the Texture dataset")
        with zipfile.ZipFile("data/textures.zip", "r") as zip_ref:
            zip_ref.extractall("data/textures")
        os.remove("data/textures.zip")
    else:
        print("Texture dataset is already downloaded.")
        print("Remove the existing folder at data/textures to download again.\n")


if __name__ == "__main__":
    main()
