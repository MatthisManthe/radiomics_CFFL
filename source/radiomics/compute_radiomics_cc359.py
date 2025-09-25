import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import json
import torch
import monai
import numpy
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, LMDBDataset, Dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    EnsureType,
    Compose
)
import glob
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from time import gmtime, strftime
import argparse
import shutil
import copy

from data.data_utils_cc359 import generate_part_data_dict, generate_random_part_data_dict
from data.data_preprocessing import generate_train_tranform, generate_val_tranform
from plot_utils import plot_train_image_pred_output, plot_labels_outputs
from models.models import modified_get_conv_layer

import radiomics
from radiomics import featureextractor
import SimpleITK as sitk
from pprintpp import pprint

import logging

from monai.transforms import (
    Transform,
    MapTransform,
    Activations,
    Activationsd,
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    LoadImage,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    RandCropByPosNegLabeld,
    Spacingd,
    SpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    ConvertToMultiChannelBasedOnBratsClassesd,
    DataStatsd,
    RandGaussianNoised,
    SqueezeDimd,
    CropForegroundd,
    SpatialPadd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    FromMetaTensord,
    SpatialResampled,
    SpatialResample,
    ScaleIntensityd
)

from collections import OrderedDict

variable_explorer = None

log_file = 'yoooooooo.txt'
handler = logging.FileHandler(filename=log_file, mode='w')  # overwrites log_files from previous runs. Change mode to 'a' to append.
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")  # format string for log messages
handler.setFormatter(formatter)
radiomics.logger.addHandler(handler)

# Control the amount of logging stored by setting the level of the logger. N.B. if the level is higher than the
# Verbositiy level, the logger level will also determine the amount of information printed to the output
radiomics.logger.setLevel(logging.DEBUG)

global_config = None
global_bin_width = None

def test_getBinEdges(parameterValues, **kwargs):
    r"""
    Calculate and return the histogram using parameterValues (1D array of all segmented voxels in the image).
  
    **Fixed bin width:**
  
    Returns the bin edges, a list of the edges of the calculated bins, length is N(bins) + 1. Bins are defined such, that
    the bin edges are equally spaced from zero, and that the leftmost edge :math:`\leq \min(X_{gl})`. These bin edges
    represent the half-open ranges of each bin :math:`[\text{lower_edge}, \text{upper_edge})` and result in gray value
    discretization as follows:
  
    .. math::
      X_{b, i} = \lfloor \frac{X_{gl, i}}{W} \rfloor - \lfloor \frac {\min(X_{gl})}{W} \rfloor + 1
  
    Here, :math:`X_{gl, i}` and :math:`X_{b, i}` are gray level intensities before and after discretization, respectively.
    :math:`{W}` is the bin width value (specfied in ``binWidth`` parameter). The first part of the formula ensures that
    the bins are equally spaced from 0, whereas the second part ensures that the minimum gray level intensity inside the
    ROI after binning is always 1.
  
    In the case where the maximum gray level intensity is equally dividable by the binWidth, i.e.
    :math:`\max(X_{gl}) \mod W = 0`, this will result in that maximum gray level being assigned to bin
    :math:`[\max(X_{gl}), \max(X_{gl}) + W)`, which is consistent with numpy.digitize, but different from the behaviour
    of numpy.histogram, where the final bin has a closed range, including the maximum gray level, i.e.
    :math:`[\max(X_{gl}) - W, \max(X_{gl})]`.
  
    .. note::
      This method is slightly different from the fixed bin size discretization method described by IBSI. The two most
      notable differences are 1) that PyRadiomics uses a floor division (and adds 1), as opposed to a ceiling division and
      2) that in PyRadiomics, bins are always equally spaced from 0, as opposed to equally spaced from the minimum
      gray level intensity.
  
    *Example: for a ROI with values ranging from 54 to 166, and a bin width of 25, the bin edges will be [50, 75, 100,
    125, 150, 175].*
  
    This value can be directly passed to ``numpy.histogram`` to generate a histogram or ``numpy.digitize`` to discretize
    the ROI gray values. See also :py:func:`binImage()`.
  
    **Fixed bin Count:**
  
    .. math::
      X_{b, i} = \left\{ {\begin{array}{lcl}
      \lfloor N_b\frac{(X_{gl, i} - \min(X_{gl})}{\max(X_{gl}) - \min(X_{gl})} \rfloor + 1 &
      \mbox{for} & X_{gl, i} < \max(X_{gl}) \\
      N_b & \mbox{for} & X_{gl, i} = \max(X_{gl}) \end{array}} \right.
  
    Here, :math:`N_b` is the number of bins to use, as defined in ``binCount``.
  
    References
  
    - Leijenaar RTH, Nalbantov G, Carvalho S, et al. The effect of SUV discretization in quantitative FDG-PET Radiomics:
      the need for standardized methodology in tumor texture analysis. Sci Rep. 2015;5(August):11075.
    """
    global global_config
    global global_bin_width

    # Copied from pyradiomics function, using min and max intensity from the whole dataset
    binWidth = global_bin_width
    lowBound = global_config["min_normalized_intensity"] - (global_config["min_normalized_intensity"] % binWidth)
    highBound = global_config["max_normalized_intensity"] + 2 * binWidth
    
    binEdges = numpy.arange(lowBound, highBound, binWidth)
    
    #print(f'Calculated {len(binEdges) - 1} bins for bin width {binWidth} with edges: {binEdges})')
    
    return binEdges  # numpy.histogram(parameterValues, bins=binedges)
  
radiomics.imageoperations.getBinEdges = test_getBinEdges


def main(args, config):
    """Main function"""
    
    # --------------------------------- With cc359 partitioning ----------------------------
    
    # Generating list of data files to load
    partition_data_dict = generate_part_data_dict(config["data_dir"], config["seg_dir"])    
    ordered_part_data_dict = OrderedDict(partition_data_dict)
    
    print(ordered_part_data_dict)
    # Define the number of clients in the federation
    nb_clients = len(partition_data_dict.keys())
    
    # Define the transformations to be applied to data
    val_transform_standard = Compose([
            # Load volumes
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            # Normalize intensity for each channel,
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            
            FromMetaTensord(keys=["image", "label"]), 
        ])
    
    val_transform_crop_pad_standard = Compose([
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CenterSpatialCropd(keys=["image", "label"], roi_size=[256, 256, -1]),
        SpatialPadd(keys=["image", "label"], spatial_size=[256, 256, -1]),
        
        # Normalize intensities for each volume,
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        
        FromMetaTensord(keys=["image", "label"])  
    ])
    
    val_transform_01norm = Compose([
            # Load volumes
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            # Normalize intensity for each channel,
            ScaleIntensityd(keys=["image"], channel_wise=True),
            
            FromMetaTensord(keys=["image", "label"]), 
        ])

    extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=0.1)
    
    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    # Initialize lists of train and validation datasets and loaders
    train_ds_list = []
    train_loader_list = []
    train_ds_list_01norm = []
    train_loader_list_01norm = []
    train_ds_list_crop_standard = []
    train_loader_list_crop_standard = []
    
    # Define train and validation datasets and dataloaders for each
    for i in ordered_part_data_dict.keys():
        train_ds_list.append(Dataset(data=ordered_part_data_dict[i], transform=val_transform_standard))
        train_ds_list_01norm.append(Dataset(data=ordered_part_data_dict[i], transform=val_transform_01norm))
        train_ds_list_crop_standard.append(Dataset(data=ordered_part_data_dict[i], transform=val_transform_crop_pad_standard))
    
    for i in range(nb_clients):
        train_loader_list.append(DataLoader(train_ds_list[i], batch_size=1, shuffle=False, num_workers=3))
        train_loader_list_01norm.append(DataLoader(train_ds_list_01norm[i], batch_size=1, shuffle=False, num_workers=3))
        train_loader_list_crop_standard.append(DataLoader(train_ds_list_crop_standard[i], batch_size=1, shuffle=False, num_workers=3))
        
    keys = list(ordered_part_data_dict.keys())
    print(keys)
    
    """
    max_norm_intensity = -200.0
    min_norm_intensity = 200.0
    
    for client in range(nb_clients):
        for (idx, batch_data) in enumerate(tqdm(train_loader_list_crop_standard[client])):
            
            inputs, labels = (
                batch_data["image"],
                batch_data["label"].int(),
            )
          
            max_norm_intensity = max(max_norm_intensity, torch.max(inputs[labels == 1]))
            min_norm_intensity = min(min_norm_intensity, torch.min(inputs[labels == 1]))
    
    print("max normalized intensity: ", max_norm_intensity)
    print("min normalized intensity: ", min_norm_intensity)
    
    # ----------------------------------------
    # Ndarray: [Batch, Channel, x, y, z]
    # Sitk image: [z, y, x, Channel, Batch]
    # ----------------------------------------
    """
    
    for bin_width in config["bin_width"]:
        
        global global_bin_width
        global_bin_width = bin_width
        
        radiomic_results_partition = {}
        
        for client in range(nb_clients):
            
            client_radiomics = []
            
            for (idx, batch_data) in enumerate(tqdm(train_loader_list_crop_standard[client])):
                
                inputs, labels = (
                    batch_data["image"],
                    batch_data["label"].int(),
                )
                
                print(inputs.shape, inputs.type())
                
                sitk_img = sitk.GetImageFromArray(inputs)
                sitk_mask = sitk.GetImageFromArray(labels)
                
                print(sitk_img.GetSize())
                result = extractor.execute(sitk_img[:,:,:,0,0], sitk_mask[:,:,:,0,0])
                pprint(result)
                
                for k in result.keys():
                    if type(result[k]).__module__ == numpy.__name__:
                        result[k] = float(result[k])
                        
                client_radiomics.append(result)
            
            radiomic_results_partition[keys[client]] = client_radiomics
                
        with open(f"radiomics_results_per_center_cc359_croppad_standard-{bin_width}.json", "w") as outfile:
            json.dump(radiomic_results_partition, outfile)
    

if __name__ == "__main__":
    
    print("Curent working directory: ", os.getcwd())
    
    print("Is cuda avaiable? ", torch.cuda.is_available())
    print("Number of cuda devices available: ", torch.cuda.device_count())
    
    print("Monai config")
    print_config()
    
    # Define argument parser and its attributes
    parser = argparse.ArgumentParser(description='Train 3D UNet on Brats')
    
    parser.add_argument('--config_path', dest='config_path', type=str,
                        help='path to json config file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Read the config file
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    global_config = config
    
    main(args, config)
