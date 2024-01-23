# ConvNeXt 
PyTorch implementation of the ["A ConvNet for the 2020s"](https://arxiv.org/pdf/2201.03545.pdf)


# Installation

We provide installation instructions for CIFAR-100 and Mini-Imagenet classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n convnext python=3.8 -y
conda activate convnext
```

Install Pytorch and torchvision following official instructions. For example:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Clone this repo and install required packages:
```
git clone https://github.com/facebookresearch/ConvNeXt
```


## Mini-Imagenet Preparation

Download Mini-Imagenet from https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view, place the pickle files into ```data/miniImagenet``` folder,

then extract files in the same folder
```
tar -xvzf data/miniImagenet/mini-imagenet.tar.gz -C data/miniImagenet
```
then run ```prepare_data.py```


## CIFAR-100 
For CIFAR-100 there is no need to download data, they will be automatically downloaded in ```data/CIFAR100 ```.


# Usage
Once you have downloaded the project and made sure you have all the requirements installed, you can train your own models simply 
running ```training.py``` if you want to train the full size models (the original ones described in the paper) or the reduced ones 
built specifically for the two datasets.



