import os
import json
import torch
from torch import nn
import lmdb
import matplotlib.pyplot as plt
import monai
import numpy as np
from monai.utils import set_determinism
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, PersistentDataset, LMDBDataset
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
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary
from time import gmtime, strftime
import argparse
import shutil
import pickle
#from torchviz import make_dot

from data.data_utils import generate_data_dict, generate_train_val_dict, random_partitioning_train_val_dict, gen_partitioning_fets
from data.data_preprocessing import ConvertToWholeTumord, generate_train_tranform, generate_val_tranform
from plot_utils import plot_train_image_pred_output, plot_labels_outputs
from models.models_reconstruction_radiomics import Decoder
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
    Resized,
    ToTensord
)


def main(args, config):
    """Main function"""
    
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
    
    nb_clients = len(train_part_list)
    clients = list(samples_inst_map_train.keys())
                   
    # Save the partitioning training / validation in the experiment folder
    with open(os.path.join(log_dir,'training_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_train, fp, indent=4)
        
    with open(os.path.join(log_dir,'validation_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_val, fp, indent=4)
    
    train_flair_list = []
    train_flair_list_to_dataset = {}
    
    val_flair_list = []
    val_flair_list_to_dataset = {}
    
    full_flair_list = []
    
    for e in train_dict:
        basename = os.path.basename(e["image"][0])
        train_flair_list.append(basename)
        full_flair_list.append(basename)
        train_flair_list_to_dataset[basename] = e
    
    for e in val_dict:
        basename = os.path.basename(e["image"][0])
        val_flair_list.append(basename)
        full_flair_list.append(basename)
        val_flair_list_to_dataset[basename] = e
        
    modalities = ["t1"]
    
    # Load radiomics files, filter train_val samples
    all_mod_part_results = {}
    for mod in modalities:
        with open(os.path.join(config["radiomic_dir"], config["part_file"].replace("#", mod)), "r") as file:
            part_results_load = json.load(file)
            all_mod_part_results[mod] = part_results_load
            
    results_t1 = all_mod_part_results["t1"]
    
    # Fix an arbitrary order for the features in a vector
    keys = list(results_t1.values())[0][0].keys()
    
    print("All keys per modality: ", keys, len(keys), "\n")
    
    final_keys = []
    for k in keys:
        if ("shape" not in k) and ("diagnostics" not in k) and (k != "id_sample"):
            final_keys.append(k)
    
    print("Final features per modality: ", final_keys, len(final_keys))
    
    # Hell, un normalized features
    dict_radiomics = {client:{} for client in clients}
    unnorm_data_matrix = []
    
    for (idm, (mod, part_results)) in enumerate(all_mod_part_results.items()):
        print(mod)
        print(len(dict_radiomics["1"]))
        for (idc, (client, v)) in enumerate(part_results.items()):
            for (idv, volume_radiomics) in enumerate(v):
                if os.path.basename(volume_radiomics["id_sample"].replace(mod, "flair")) in full_flair_list:
                    if idm == 0:
                        features = []
                        for (idx, k) in enumerate(final_keys):
                            features.append(volume_radiomics[k])
                        basename = os.path.basename(volume_radiomics["id_sample"].replace(mod, "flair"))
                        unnorm_data_matrix.append(features)
                        dict_radiomics[client][basename] = features
    
    unnorm_data_matrix = np.array(unnorm_data_matrix)
    print("Unnorm data matrix shape: ", unnorm_data_matrix.shape)
    
    # Compute normalization function
    min_scale_values_per_key = np.percentile(unnorm_data_matrix, 2, axis=0)
    max_scale_values_per_key = np.percentile(unnorm_data_matrix, 98, axis=0)
    print("Length min scale values per key: ", len(min_scale_values_per_key))
    
    #print(len(dict_radiomics["1"]), len(dict_radiomics["1"]["FeTS2022_01377_flair.nii.gz"]))
        
    # Normalized features
    for client in dict_radiomics.keys():
        for (sample, features) in dict_radiomics[client].items():
            normalized_features = []
            for idf in range(len(features)):
                value = features[idf]
                value = (value - min_scale_values_per_key[idf])/(max_scale_values_per_key[idf] - min_scale_values_per_key[idf])
                value = max(value, 0.0)
                value = min(1.0, value)
                normalized_features.append(value)
            dict_radiomics[client][sample] = normalized_features
       
    train_set = []
    val_set = []
    
    for client in dict_radiomics.keys():
        for (sample, norm_features) in dict_radiomics[client].items():
            if sample in train_flair_list:
                train_set.append({"image":norm_features, "label":train_flair_list_to_dataset[sample]["image"][1]})
            elif sample in val_flair_list:
                val_set.append({"image":norm_features, "label":val_flair_list_to_dataset[sample]["image"][1]})
    
    print(train_set)
    
    train_transform = Compose([
        
          LoadImaged(keys=["label"]),
          
          ToTensord(keys=["image"], dtype=torch.float),
          
          EnsureChannelFirstd(keys=["label"]),
          
          Resized(keys=["label"], spatial_size=config["size"]),
          
          # Normalize intensity for each channel,
          NormalizeIntensityd(keys=["label"], nonzero=False, channel_wise=True),
          
          FromMetaTensord(keys=["label"])
    ])
    
    val_transform = train_transform
    
    print(train_transform(train_set[0]))
    
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    
    # Define model, loss function, optimizer and metric
    model = Decoder().to(device)
     
    for name, param in model.named_parameters():
        print(name, param.shape)

    # Define train and validation datasets and dataloaders
    if config["persistent_dataset"]:
        # Delete every cached elements
        cache_dir = config["cache_dir"]
        filelist = glob.glob(os.path.join(cache_dir, "*"))
        for f in filelist:
            os.remove(f)
            
        train_ds = LMDBDataset(data=train_dict, transform=train_transform, cache_dir=config["cache_dir"])
        val_ds = LMDBDataset(data=val_dict, transform=val_transform, cache_dir=config["cache_dir"])
    else:
        train_ds = CacheDataset(data=train_set, transform=train_transform)
        val_ds = CacheDataset(data=val_set, transform=val_transform)
        
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0)    
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(model, next(iter(train_loader))["image"].to(device))
    print(summary(model, next(iter(train_loader))["image"].to(device), show_input=False, show_hierarchical=True))
    #make_dot(model(next(iter(train_loader))["image"].to(device)), params=dict(list(model.named_parameters()))).render("fuck_monai_net", format="png")

    # Define max number of training epoch
    max_epochs = config["max_epochs"]
    
    # Define loss function, optimizer and metrics
    loss_function = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
    
    metric = nn.MSELoss()
        
    # Initialize metrics
    best_mse = 100000000.0
    best_epoch = -1
    
    # Training process
    for epoch in range(config["max_epochs"]):
        
        # Setting initial time, step and loss variables
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        
        model.train()
        
        epoch_loss = 0
        step = 0
        
        # Main loop on training set
        for (idx, batch_data) in enumerate(tqdm(train_loader)):
            step += 1
            
            # Get data from batch
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Print some images at the begining of training
            if epoch==0 and idx<5 and False:
                with torch.no_grad():
                    fig = plot_train_image_pred_output(inputs, outputs, labels)
                    writer.add_figure("Initial check", fig, epoch+1, close=False)
                    plt.show()
                    plt.close()
                
            # Backward: Evaluate loss, backward and step from optimizer
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Get loss value on last batch
            epoch_loss += loss.item()*inputs.shape[0]      
        
        # lr_scheduler.step()
        epoch_loss /= len(train_ds)
        
        # Add loss value to tensorboard, and print it
        writer.add_scalar("Loss/train", epoch_loss, epoch+1)
        print(f"\nepoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
        # Validation each val_interval number of epochs
        if (epoch) % config["validation_interval"] == 0:
            
            model.eval()
            
            average_mse = 0
            
            with torch.no_grad():
                
                # Main loop on validation set
                for (idx, val_data) in enumerate(tqdm(val_loader)):
                    
                    # Get data from batch
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    
                    # Sliding window inference keeping the same roi size as in training
                    val_outputs = model(val_inputs)
     
                    # Plot results each print_val_interval number of training epochs
                    if (epoch + 1) % config['print_validation_interval'] == 0:
                        slice_id = 32 + np.random.randint(64)
                        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20)) 
                                    
                        # Show a slice of input data
                        plt.subplot(1, 2, 1)
                        plt.title("target", fontsize=20)
                        plt.imshow(val_labels[0, 0, :, :, slice_id].cpu(), cmap="gray")
                        plt.colorbar()
                        
                        plt.subplot(1, 2, 2)
                        plt.title("prediction", fontsize=20)
                        plt.imshow(val_outputs[0, 0, :, :, slice_id].cpu(), cmap="gray")
                        plt.colorbar()
                        
                        writer.add_figure(f"Validation plot/{epoch+1}", fig, idx, close=False)
                        plt.show()
                        plt.close()
                    
                    # compute metric for current iteration
                    value = metric(val_outputs, val_labels)
                    average_mse += value*val_inputs.shape[0]
                
                # Aggregate the final mean dice result
                metric_d = average_mse/len(val_ds)
                
                # Add these metrics to tensorboard
                writer.add_scalar("Evaluation/MSE", metric_d, epoch+1)
                
                # Save best model (based on mean dice metric)
                if metric_d < best_mse:
                    best_mse = metric_d
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
                    print("\nSaved new best model")
                # Print metrics after each epoch
                print(
                    f"\nCurrent epoch: {epoch + 1}\nCurrent MSE: {metric_d:.4f}"
                    f"\nBest mean dice: {best_mse:.4f}"
                    f" at epoch: {best_epoch}"
                )
                
    # Adding hyperparameters value to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
    writer.add_hparams(config_hparam, {"hparam/Mean_dice":best_mse,
                                       "hparam/Best_epoch":best_epoch})

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
    
    main(args, config)
