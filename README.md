# Whole-brain radiomics for clustered federated personalization in brain tumor segmentation

This is the official code base for the paper 
> [Matthis Manthe et al., “Whole Brain Radiomics for Clustered Federated Personalization in Brain Tumor Segmentation,” Medical Imaging with Deep Learning, PMLR, January 23, 2024](https://proceedings.mlr.press/v227/manthe24a.html)

Due to the extremely short time of implementation (the crunch was real), this is probably one of the worst code bases I have ever written. One can dive into this code to clarify uncertainties when reading the paper, but I would strongly advise to reimplement the method in your own code base using the presented logic if needed. This is not a plug-and-play code for other datasets than the ones in the experiments of the paper.

## Code base structure
The structure of the code base is simple
- Directly in the ```/source``` folder are five python codes, one for each training algorithm tested in the paper (Centralized, FedAvg, Local finetuning, CFFL Ideal and CFFL),
- In the directory ```/source/radiomics``` are the two python codes used to compute radiomic features on CC359 and FeTS datasets. These radiomic features are output in a single json file.
- In each directory is a ```*/config``` directory storing config files as json files.

## Launching an experiment
All these python files can be ran using the following typical command

```python3 NAME.py --config_path config/CONFIG_NAME.json```

which, for a centralized training following the configuration defined in *config_centralized.json*, becomes 

```python3 Centralized.py --config_path config/config_centralized.json```

An example of a json config file for each code can be found in ```*/config``` folders, storing every selected hyperparameters.

One needs to create a ```/runs``` folder for experiment folders to be created every time a code is ran, containing everything related to this experiment instance (tensorboard, model weights, copy of the config file, etc.).

## Dependencies
The main frameworks used are essentially 
- Pytorch
- Numpy
- Sklearn
- Monai
- Pyradiomics
- SimpleITK

with additional dependencies with tqdm, glob, pandas, pickle and pacmap.

