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

from data.data_utils import generate_data_dict, gen_partitioning_fets
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

global_config = None
global_bin_width = None
modality = None

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
    global modality

    # Copied from pyradiomics function, using min and max intensity from the whole dataset
    binWidth = global_bin_width
    lowBound = global_config["min_normalized_intensity"][modality] - (global_config["min_normalized_intensity"][modality] % binWidth)
    highBound = global_config["max_normalized_intensity"][modality] + 2 * binWidth
    
    binEdges = numpy.arange(lowBound, highBound, binWidth)
    
    #print(f'Calculated {len(binEdges) - 1} bins for bin width {binWidth} with edges: {binEdges})')
    
    return binEdges  # numpy.histogram(parameterValues, bins=binedges)
  


def main(args, config):
    """Main function"""
    
    log_file = f'FeTS2022_radiomics_log_bin_width_{config["bin_width"]}_modality_{config["modality"]}_mask_{config["mask"]}.txt'
    handler = logging.FileHandler(filename=log_file, mode='w')  # overwrites log_files from previous runs. Change mode to 'a' to append.
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")  # format string for log messages
    handler.setFormatter(formatter)
    radiomics.logger.addHandler(handler)
    
    # Control the amount of logging stored by setting the level of the logger. N.B. if the level is higher than the
    # Verbositiy level, the logger level will also determine the amount of information printed to the output
    radiomics.logger.setLevel(logging.DEBUG)

    # --------------------------------- With FeTS2022 partitioning ----------------------------
    
    # Initializing tensorboard summary
    log_dir = config["log_dir"] + "/" + os.path.basename(__file__)
    if config["log_filename"] is not None:
        log_dir += "/"+config["log_filename"]
    else:
        log_dir += "/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    writer = SummaryWriter(log_dir=log_dir)
    
    # Copy config file in the experiment folder
    shutil.copy(args.config_path, log_dir)
    
    # Generating list of data files to load
    data_dir = config['data_dir']
    data_dict = generate_data_dict(data_dir)
    
    # Loading training and validation sets
    train_part_list, val_part_list, train_dict, val_dict, \
        samples_inst_map_train, samples_inst_map_val = gen_partitioning_fets(data_dict, 
                                                                                config["partition_file"], 
                                                                                prop_full_dataset=config['prop_full_dataset'], 
                                                                                ratio_train=config['ratio_train'])
    # Save the partitioning training / validation in the experiment folder
    with open(os.path.join(log_dir,'training_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_train, fp, indent=4)
        
    with open(os.path.join(log_dir,'validation_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_val, fp, indent=4)
    
    # Define the number of clients in the federation
    nb_clients = len(samples_inst_map_train.keys())
    
    # Define the transformations to be applied to data
    val_transform = generate_val_tranform(roi_size=config["roi_size"])

    if config["histogram_computation"] == "same_hist_fixed_bin_width":
        radiomics.imageoperations.getBinEdges = test_getBinEdges
        extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=0.1)
    elif config["histogram_computation"] == "fixed_bin_width":
        extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=config["bin_width"])
    elif config["histogram_computation"] == "fixed_bin_count":
        extractor = featureextractor.RadiomicsFeatureExtractor(binCount=config["bin_count"])
    
    print('Extraction parameters:\n\t', extractor.settings)
    print('Enabled filters:\n\t', extractor.enabledImagetypes)
    print('Enabled features:\n\t', extractor.enabledFeatures)

    # Initialize lists of train and validation datasets and loaders
    train_ds_list = []
    train_loader_list = []
    
    # Define train and validation datasets and dataloaders for each
    for i in range(nb_clients):
        train_ds_list.append(Dataset(data=train_part_list[i], transform=val_transform))
        
    for i in range(nb_clients):
        train_loader_list.append(DataLoader(train_ds_list[i], batch_size=1, shuffle=False, num_workers=3))
        
    keys = list(samples_inst_map_train.keys())
    print(keys)
    
    """
    max_norm_intensity = [-200.0,-200.0, -200.0, -200.0]
    min_norm_intensity = [200.0, 200.0, 200.0, 200.0]
    
    for client in range(nb_clients):
        for (idx, batch_data) in enumerate(tqdm(train_loader_list[client])):
            
            inputs, labels = (
                batch_data["image"],    
                batch_data["label"].int(),
            )
          
            s = 120
            roi = labels[:,1,:,:,:]
            mod_volume = inputs[:,0,:,:,:]
            
            if config["mask"] == "brain":
                brain_wo_tumor_mask = mod_volume>mod_volume.min()
            elif config["mask"] == "tumor":
                brain_wo_tumor_mask = roi==1
            elif config["mask"] == "brain without tumor":
                brain_wo_tumor_mask = torch.logical_and(mod_volume>mod_volume.min(), roi == 0)
            
            for m in range(4):
                
                plt.imshow(inputs[0,m,:,:,s], cmap="gray")
                plt.show()
                plt.close()
                plt.imshow(inputs[0,m,:,:,s], cmap="gray")
                plt.imshow(brain_wo_tumor_mask[0,:,:,s], alpha=brain_wo_tumor_mask[0,:,:,s]*0.2)
                plt.show()
                plt.close()
                
                print(torch.max(inputs[:,m,:,:,:][brain_wo_tumor_mask]))
                print(torch.min(inputs[:,m,:,:,:][brain_wo_tumor_mask]))
                max_norm_intensity[m] = max(max_norm_intensity[m], torch.max(inputs[:,m,:,:,:][brain_wo_tumor_mask]))
                min_norm_intensity[m] = min(min_norm_intensity[m], torch.min(inputs[:,m,:,:,:][brain_wo_tumor_mask]))

    
    print("max normalized intensity: ", max_norm_intensity)
    print("min normalized intensity: ", min_norm_intensity)
    
    # ----------------------------------------
    # Ndarray: [Batch, Channel, x, y, z]
    # Sitk image: [z, y, x, Channel, Batch]
    # ----------------------------------------
    """
    bin_width = config["bin_width"]
    
    modalities = ["flair", "t1", "t1ce", "t2"]
    
    global global_bin_width
    global modality
    global_bin_width = bin_width
    
    radiomic_results_partition = {}
    
    for client in range(nb_clients):
        
        client_radiomics = []
        
        for (idx, batch_data) in enumerate(tqdm(train_loader_list[client])):
            
            adr = train_part_list[client][idx]
            
            inputs, labels = (
                batch_data["image"],
                batch_data["label"].int(),
            )
            
            roi = labels[0,1,:,:,:]
            mod_volume = inputs[0,0,:,:,:]
            
            if config["mask"] == "brain":
                brain_wo_tumor_mask = mod_volume>mod_volume.min()
            elif config["mask"] == "tumor":
                brain_wo_tumor_mask = roi==1
            elif config["mask"] == "brain without tumor":
                brain_wo_tumor_mask = torch.logical_and(mod_volume>mod_volume.min(), roi == 0)
            
            m = config["modality"]
                
            modality = m
            print(inputs.shape)
            sitk_img = sitk.GetImageFromArray(inputs[0,m,:,:,:])
            sitk_mask = sitk.GetImageFromArray(brain_wo_tumor_mask.int())
            
            print(sitk_img.GetSize(), sitk_mask.GetSize())
            
            result = extractor.execute(sitk_img, sitk_mask)
            pprint(result)
            
            for k in result.keys():
                if type(result[k]).__module__ == numpy.__name__:
                    result[k] = float(result[k])
                 
            result["id_sample"] = adr["image"][m]
            
            client_radiomics.append(result)
        
        radiomic_results_partition[keys[client]] = client_radiomics
            
    with open(f"radiomics_results_per_center_FeTS2022_cropforeground_standard-{bin_width}_modality_{modalities[m]}_mask_{config['mask']}.json", "w") as outfile:
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
