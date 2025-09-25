import os
import numpy as np
import glob
import pandas as pd
import json
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_part_data_dict(data_dir, seg_dir):

    scanners = ["philips_3", "philips_15", "siemens_3", "siemens_15", "ge_3", "ge_15"]
    partition_data = dict.fromkeys(scanners)

    for scan in scanners:
        images = sorted(
            glob.glob(os.path.join(data_dir, "*"+scan+"*.nii.gz"))
        )
        seg = sorted(
            glob.glob(os.path.join(seg_dir, "*"+scan+"*.nii.gz"))
        )
        data_dict = [
            {"image": image_name, "label":seg_name}
            for image_name, seg_name in zip(images, seg)
        ]
        partition_data[scan] = data_dict
    
    return partition_data


def generate_part_data_dict_train_val_test(data_dir, seg_dir, train_split_file, val_split_file, test_split_file):
    
    with open(train_split_file, 'r') as f:
        train_split = json.load(f)
    with open(val_split_file, 'r') as f:
        val_split = json.load(f)
    with open(test_split_file, 'r') as f:
        test_split = json.load(f)
    
    images = sorted(
        glob.glob(os.path.join(data_dir, "*.nii.gz"))
    )
    segs = sorted(
        glob.glob(os.path.join(seg_dir, "*.nii.gz"))
    )

    train = {}
    val = {}
    test = {}
    
    scanners = ["philips_3", "philips_15", "siemens_3", "siemens_15", "ge_3", "ge_15"]
    for scan in scanners:
        # Train dict
        data_dict = []
        for patient in train_split[scan]:
            scan_image = sorted(
                [image for image in images if patient in image]
            )
            scan_seg = sorted(
                [seg for seg in segs if patient in seg]
            )
            data_dict += [
                {"image": image_name, "label":seg_name}
                for image_name, seg_name in zip(scan_image, scan_seg)
            ]
        train[scan] = data_dict
        # Val dict    
        data_dict = []
        for patient in val_split[scan]:
            scan_image = sorted(
                [image for image in images if patient in image]
            )
            scan_seg = sorted(
                [seg for seg in segs if patient in seg]
            )
            data_dict += [
                {"image": image_name, "label":seg_name}
                for image_name, seg_name in zip(scan_image, scan_seg)
            ]
        val[scan] = data_dict
  
        # Test dict    
        data_dict = []
        for patient in test_split[scan]:
            scan_image = sorted(
                [image for image in images if patient in image]
            )
            scan_seg = sorted(
                [seg for seg in segs if patient in seg]
            )
            data_dict += [
                {"image": image_name, "label":seg_name}
                for image_name, seg_name in zip(scan_image, scan_seg)
            ]
        test[scan] = data_dict

    return train, val, test
    
    
# Geenrate a dict with only the file names for each center
def generate_patient_part_dict(data_dir):
    
    scanners = ["philips_3", "philips_15", "siemens_3", "siemens_15", "ge_3", "ge_15"]
    partition_data = dict.fromkeys(scanners)
    
    for scan in scanners:
        images = sorted(
            glob.glob(os.path.join(data_dir, "*"+scan+"*.nii.gz"))
        )
        
        data_dict = [os.path.basename(image_name).split('.')[0] for image_name in images]
        
        partition_data[scan] = data_dict
     
    return partition_data
     
  
def generate_3D_dict(data_dir, seg_dir, split_file):
    
    with open(split_file, 'r') as f:
        split = json.load(f)

    images = sorted(
        glob.glob(os.path.join(data_dir, "*.nii.gz"))
    )
    segs = sorted(
        glob.glob(os.path.join(seg_dir, "*.nii.gz"))
    )

    result = {}
    
    scanners = ["philips_3", "philips_15", "siemens_3", "siemens_15", "ge_3", "ge_15"]
    for scan in scanners:
        # Train dict
        data_dict = []
        for patient in split[scan]:
            scan_image = sorted(
                [image for image in images if patient in image]
            )
            scan_seg = sorted(
                [seg for seg in segs if patient in seg]
            )
            data_dict += [
                {"image": image_name, "label":seg_name}
                for image_name, seg_name in zip(scan_image, scan_seg)
            ]
        result[scan] = data_dict
        
    return result


def generate_random_part_data_dict(data_dir, seg_dir, seed=0):
    
    images = sorted(
        glob.glob(os.path.join(data_dir, "*.nii.gz"))
    )
    
    seg = sorted(
        glob.glob(os.path.join(seg_dir, "*.nii.gz"))
    )
    
    np.random.seed(seed)
    shuffle = np.arange(len(images))
    np.random.shuffle(shuffle)
    
    random_clients = ["0", "1", "2", "3", "4", "5"]
    
    partition_data = dict.fromkeys(random_clients)
    
    for (idx, client) in enumerate(random_clients):
        
        part_images = [images[i] for i in shuffle[60*idx:60*(idx+1)]]
        part_seg = [seg[i] for i in shuffle[60*idx:60*(idx+1)]]
        
        data_dict = [
            {"image":image_name, "label":seg_name}
            for image_name, seg_name in zip(part_images, part_seg)
        ]
        partition_data[client] = data_dict
        
    return partition_data


# Standardize/normalize each volume and save each axial slice in a separate nifti file for 2D training.
def preprocess_slices_cc359(data_dir, seg_dir, image_output_dir, seg_output_dir):
    
    images = sorted(
        glob.glob(os.path.join(data_dir, "*.nii.gz"))
    )
    
    segs = sorted(
        glob.glob(os.path.join(seg_dir, "*.nii.gz"))
    )
    
    for (image, seg) in tqdm(zip(images, segs), total=len(images)):
        loaded_image = nib.load(image)
        loaded_seg = nib.load(seg)

        image_array = loaded_image.get_fdata()
        norm_image = (image_array - image_array.min())/(image_array.max() - image_array.min())

        for slice_nb in range(norm_image.shape[2]):
            
            slice_image = nib.Nifti1Image(norm_image[:,:,slice_nb].astype(np.float32), loaded_image.affine, header=loaded_image.header)
            slice_seg = nib.Nifti1Image(loaded_seg.get_fdata()[:,:,slice_nb].astype(np.float32), loaded_seg.affine, header=loaded_seg.header)
            
            nib.save(slice_image, os.path.join(image_output_dir, str(slice_nb)+"_"+os.path.basename(image)))
            nib.save(slice_seg, os.path.join(seg_output_dir, str(slice_nb)+"_"+os.path.basename(seg)))
            
"""
data_dir = "/home/manthe/Documents/PhD_works/phd-first-centralized-trainings/datasets/cc359_preprocessed/Registered"
seg_dir = "/home/manthe/Documents/PhD_works/phd-first-centralized-trainings/datasets/cc359_preprocessed/Registered_STAPLE"
image_output_dir = "/home/manthe/Documents/PhD_works/phd-first-centralized-trainings/datasets/cc359_preprocessed/Registered_01norm_slices"
seg_output_dir = "/home/manthe/Documents/PhD_works/phd-first-centralized-trainings/datasets/cc359_preprocessed/Registered_STAPLE_01norm_slices"
preprocess_slices_cc359(data_dir, seg_dir, image_output_dir, seg_output_dir)  
"""    


