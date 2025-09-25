import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from monai.utils import first, set_determinism
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, Dataset, CacheDataset, PersistentDataset
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import nibabel
from pprint import pprint
import glob


def plot_train_image_pred_output(inputs, outputs, labels):

    slice_id = 32 + np.random.randint(64)
    
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(45, 40)) 
                
    # Show a slice of input data
    plt.subplot(4, 4, 1)
    plt.title("image flair", fontsize=20)
    im = axes.flat[0].imshow(inputs[0, 0, :, :, slice_id].cpu(), cmap="gray")
    plt.subplot(4, 4, 2)
    plt.title("image T1", fontsize=20)
    axes.flat[1].imshow(inputs[0, 1, :, :, slice_id].cpu(), cmap="gray")
    plt.subplot(4, 4, 3)
    plt.title("image T1ce", fontsize=20)
    axes.flat[2].imshow(inputs[0, 2, :, :, slice_id].cpu(), cmap="gray")
    plt.subplot(4, 4, 4)
    plt.title("image T2", fontsize=20)
    axes.flat[3].imshow(inputs[0, 3, :, :, slice_id].cpu(), cmap="gray")

    # Show the outputs and labels for this slice
    for i in range(3):
        name_label = "Tumor core" if i==0 else "Whole tumor" if i==1 else "Enhancing tumor"
        plt.subplot(4, 4, 4*(i+1) + 1)
        plt.title(f"outputs {name_label}", fontsize=20)
        axes.flat[4*(i+1)].imshow(outputs[0, i, :, :, slice_id].cpu(), cmap="gray")
        plt.subplot(4, 4, 4*(i+1) + 2)
        plt.title(f"label {name_label}", fontsize=20)
        axes.flat[4*(i+1) + 1].imshow(labels[0, i, :, :, slice_id].cpu())
        plt.subplot(4, 4, 4*(i+1) + 3)
        plt.title(f"highlight ground truth {name_label}", fontsize=20)
        axes.flat[4*(i+1) + 2].imshow(inputs[0, 0, :, :, slice_id].cpu(), cmap="gray")
        axes.flat[4*(i+1) + 2].imshow(labels[0, i, :, :, slice_id].cpu(), cmap="jet", alpha=0.4)
        plt.subplot(4, 4, 4*(i+1) + 4)
        plt.title(f"highlight prediction {name_label}", fontsize=20)
        axes.flat[4*(i+1) + 3].imshow(inputs[0, 0, :, :, slice_id].cpu(), cmap="gray")
        axes.flat[4*(i+1) + 3].imshow(outputs[0, i, :, :, slice_id].cpu(), cmap="jet", alpha=0.4)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.125, 0.05, 0.7])
    cbar_ax.tick_params(labelsize=20)
    fig.colorbar(im, cax=cbar_ax)

    return fig
    
    
def plot_labels_outputs(inputs, outputs, labels):
    
    slice_id = 32 + np.random.randint(64)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(40, 30))

    for i in range(3):
        name_label = "Tumor core" if i==0 else "Whole tumor" if i==1 else "Enhancing tumor"

        plt.subplot(3, 4, 4*i+1)
        plt.title("image", fontsize=20)
        plt.imshow(inputs[0, 0, :, :, slice_id].cpu(), cmap="gray")
        plt.subplot(3, 4, 4*i+2)
        plt.title(f"outputs {name_label}", fontsize=20)
        plt.imshow(outputs[0][i, :, :, slice_id].cpu(), cmap="gray")
        plt.subplot(3, 4, 4*i+3)
        plt.title(f"label {name_label}", fontsize=20)
        plt.imshow(labels[0, i, :, :, slice_id].cpu())
        plt.subplot(3, 4, 4*i+4)
        plt.title(f"highlight {name_label}", fontsize=20)
        plt.imshow(outputs[0][i, :, :, slice_id].cpu(), cmap="gray")
        plt.imshow(labels[0, i, :, :, slice_id].cpu(), cmap="jet", alpha=0.5)
    
    return fig