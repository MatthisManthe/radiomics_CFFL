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
    ToTensord
)


class ConvertToWholeTumord(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor). We only convert to Whole tumor, the "or" 
    of every class labels.

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # merge labels 1, 2 and 4 to construct WT
            result = np.logical_or(
                        np.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                    )
            d[key] = np.stack(result, axis=0).astype(np.float32)
            # print(d[key][d[key]>0])
        return d

"""
# Old version (Commented 06/03/2023)
def generate_train_tranform(roi_size=[128, 128, 128], data_aug=False, device=None):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Chose segmentation map to the three labels of the BraTS challenge,
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensity for each channel,
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        
        # Randomly crop the image to crops of 128*128*128 as in SOTA papers,
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False)
        
    ]
    
    # Pushing data to GPU after one fast CPU random transformation so
    # CacheDataset won't load every sample in the GPU memory (not enough space), but will 
    # reuse on the fly some space on GPU to perform expensive transforms
    if device is not None and data_aug:
        transforms += [EnsureTyped(keys=["image", "label"], device=device, track_meta=False)]
    else:
        transforms += [FromMetaTensord(keys=["image", "label"])]
                                       
    # Data augmentations
    if data_aug:
        transforms += [
            # Random Gaussian Noise
            RandGaussianNoised(keys=['image'], prob=0.15, std=0.33),
            
            # Random Gaussian Smoothing (gaussian kernel)
            RandGaussianSmoothd(keys=['image'], prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            
            # Random intensity scaling
            RandScaleIntensityd(keys=['image'], prob=0.15, factors=0.3),
            
            # Random Contrast adjustment
            RandAdjustContrastd(keys=['image'], prob=0.15),
            
            # Avoid recasting as MetaTensor from monai > 0.9.0
            FromMetaTensord(keys=["image"])
        ]

    return Compose(transforms)
"""

def generate_train_tranform(roi_size=[128, 128, 128], data_aug=False, device=None):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Chose segmentation map to the three labels of the BraTS challenge,
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensity for each channel,
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        
        # Randomly crop the image to crops of 128*128*128 as in SOTA papers,
        RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False)
    ]
    
    # Pushing data to GPU after one fast CPU random transformation so
    # CacheDataset won't load every sample in the GPU memory (not enough space), but will 
    # reuse on the fly some space on GPU to perform expensive transforms
    #if device is not None and data_aug:
        #transforms += [EnsureTyped(keys=["image", "label"], device=device, track_meta=False)]
    if not data_aug:
        transforms += [ToTensord(keys=["image", "label"], device=device, track_meta=False)]
                                       
    # Data augmentations
    if data_aug:
        transforms += [
            
            # Push to device if specified
            EnsureTyped(keys=["image", "label"], device=device),
            
            # Random flip on each axis
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=-1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=-2),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=-3),
            
            # Random Gaussian Noise
            RandGaussianNoised(keys=['image'], prob=0.15, std=0.25),
            
            # Random Gaussian Smoothing (gaussian kernel)
            RandGaussianSmoothd(keys=['image'], prob=0.15, sigma_x=(0.5, 1.), sigma_y=(0.5, 1.), sigma_z=(0.5, 1.)),
            
            # Random intensity scaling
            RandScaleIntensityd(keys=['image'], prob=0.15, factors=0.3),
            
            # Random Contrast adjustment
            RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.5, 2)),
            
            # Avoid recasting as MetaTensor from monai > 0.9.0
            #FromMetaTensord(keys=["image", "label"]) -> Fonction de merde
            ToTensord(keys=["image", "label"], track_meta=False)
        ]

    return Compose(transforms)


def generate_val_tranform(roi_size=[128, 128, 128]):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Chose segmentation map to the three labels of the BraTS challenges,
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        
        # Normalize intensity for each channel
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        
        # Avoid using MetaTensor from monai > 0.9.0
        FromMetaTensord(keys=["image", "label"])
    ]
    
    return Compose(transforms)


def generate_inference_tranform(roi_size=[128, 128, 128]):
    
    transforms = [
        # Load volumes
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        
        # Crop to remove a maximum of the background and pad to fit the min crop size
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=roi_size),
        
        # Normalize intensity for each channel
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        
        # Avoid using MetaTensor from monai > 0.9.0
        FromMetaTensord(keys=["image"])
    ]
    
    return Compose(transforms)


def generate_autoencoder_transform(roi_size=[192, 192, 192]):
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            
            # Crop to remove a maximum of the background and pad to fit the min roi size
            CropForegroundd(keys=["image"], source_key="image"),
            CenterSpatialCropd(keys=["image"], roi_size=roi_size),
            SpatialPadd(keys=["image"], spatial_size=roi_size),
            
            # Normalize intensity for each channel,
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ]
    )
    
    return transform


def generate_autoencoder_2D_transform():
    transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"])
        ]
    )
    
    return transform