import numpy as np
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
    ScaleIntensityd
)

# ------------------- Crop-pad standardization ---------------
def generate_pre_train_3D_tranform(roi_size=[256, 256, -1]):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensities for each volume,
        NormalizeIntensityd(keys=["image"])
    ]

    return Compose(transforms)


def generate_train_2D_transform(data_aug=False):
    return Compose(FromMetaTensord(keys=["image", "label"]))
    
    
def generate_val_3D_tranform(roi_size=[256, 256, -1]):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensities for each volume,
        NormalizeIntensityd(keys=["image"]),
        
        FromMetaTensord(keys=["image", "label"])
    ]
    
    return Compose(transforms)


# -------------------- 01 norm ----------------------
def generate_pre_train_3D_tranform_01(roi_size=[256, 256, -1]):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensity for each channel,
        ScaleIntensityd(keys=["image"], channel_wise=True)
    ]

    return Compose(transforms)  
    
def generate_val_3D_tranform_01(roi_size=[256, 256, -1]):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensity for each channel,
        ScaleIntensityd(keys=["image"], channel_wise=True),
        
        FromMetaTensord(keys=["image", "label"])
    ]
    
    return Compose(transforms)

