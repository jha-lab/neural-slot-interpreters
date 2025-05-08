#!/bin/sh

# module load anaconda3/2020.11
conda create --name nsi pytorch torchvision torchaudio cudatoolkit=11.1 --channel pytorch --channel nvidia

conda activate nsi

# Add other packages and enabling extentions
conda install -c conda-forge tqdm ipywidgets matplotlib scikit-optimize numpy

#install timm
conda install -c conda-forge timm

#install h5py
conda install -c conda-forge h5py

#Install tensorboard
conda install -c conda-forge tensorboard

