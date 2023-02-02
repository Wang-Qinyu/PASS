# Mesophilic Argonaute-mediated Polydisperse Droplets Biosensor for Amplification-free, One-pot, and Multiplexed Nucleic Acid Detection using Deep Learning
## Operation Environment
Recommended linux operating system with GPU devices. The systems and equipment we use are:
- Ubuntu 18.04.2
- NVIDIA GeForce RTX 3090

## Installation 
First install Detectron2 following the official guide: [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)



## Preparations
Your directory structure should be consistent with the following:
```bash
.
├─datasets
│  ├─images
│  │  ├─test
│  │  └─train
│  └─labels
└─detectCode
```
## Training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Get the performance of the model
```bash
CUDA_VISIBLE_DEVICES=0 python eval.py
```