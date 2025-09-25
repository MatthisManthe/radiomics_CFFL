import os
import json
import torch
import lmdb
import matplotlib.pyplot as plt
import monai
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
#from torchviz import make_dot

from data.data_utils import generate_data_dict, generate_train_val_dict, random_partitioning_train_val_dict, gen_partitioning_fets
from data.data_preprocessing import ConvertToWholeTumord, generate_train_tranform, generate_val_tranform
from plot_utils import plot_train_image_pred_output, plot_labels_outputs
from models.models import modified_get_conv_layer


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
    
    # Save the partitioning training / validation in the experiment folder
    with open(os.path.join(log_dir,'training_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_train, fp, indent=4)
        
    with open(os.path.join(log_dir,'validation_samples.json'), 'w') as fp:
        json.dump(samples_inst_map_val, fp, indent=4)
        
    
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    
    # Define model, loss function, optimizer and metric

    monai.networks.blocks.dynunet_block.get_conv_layer = modified_get_conv_layer
    model = DynUNet(
        spatial_dims = 3,
        in_channels = 4,
        out_channels = 3,
        kernel_size = (3, 3, 3, 3),
        filters = (16, 32, 64, 128),
        strides = (1, 2, 2, 2),
        upsample_kernel_size = (2, 2, 2),
        norm_name=(config["norm"], {"affine": config["learn_affine_norm"]} if config["norm"] != "GROUP" else {"affine":config["learn_affine_norm"], "num_groups":1}),
        act_name=("LeakyReLu", {"negative_slope":0.01}),
        trans_bias=True,
    ).to(device)
     
    for name, param in model.named_parameters():
        print(name, param.shape)
      
    # Define the transformations to be applied to data
    train_transform = generate_train_tranform(roi_size=config["roi_size"], data_aug=config["data_augmentation"], device=device)
    val_transform = generate_val_tranform(roi_size=config["roi_size"])
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )

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
        train_ds = CacheDataset(data=train_dict, transform=train_transform)
        val_ds = CacheDataset(data=val_dict, transform=val_transform)
        
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)    
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(model, next(iter(train_loader))["image"].to(device))
    print(summary(model, next(iter(train_loader))["image"].to(device), show_input=False, show_hierarchical=True))
    #make_dot(model(next(iter(train_loader))["image"].to(device)), params=dict(list(model.named_parameters()))).render("fuck_monai_net", format="png")

    # Define max number of training epoch
    max_epochs = config["max_epochs"]
    
    # Define loss function, optimizer and metrics
    loss_function = DiceLoss(sigmoid=True, smooth_nr=1, smooth_dr=1, squared_pred=False)
    
    optimizer = None
    if config["optim"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=config["lr_factor"], patience=config["patience"], verbose=True, min_lr=config["min_lr"])
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
        
    # Initialize metrics
    best_mean_dice = -1
    best_dice_et = -1
    best_dice_tc = -1
    best_dice_wt = -1
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
            
            with torch.no_grad():
                
                # Main loop on validation set
                for (idx, val_data) in enumerate(tqdm(val_loader)):
                    
                    # Get data from batch
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    
                    # Sliding window inference keeping the same roi size as in training
                    val_outputs = sliding_window_inference(val_inputs, config["roi_size"], config["batch_size"], model, overlap=config["sliding_window_overlap"], mode="gaussian")
                    
                    # Thresholding at 0.5 to get segmentation map
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
     
                    # Plot results each print_val_interval number of training epochs
                    if (epoch + 1) % config['print_validation_interval'] == 0 and False:
                        fig = plot_labels_outputs(val_inputs, val_outputs, val_labels)
                        writer.add_figure(f"Validation plot/{epoch+1}", fig, epoch+1, close=False)
                        plt.show()
                    
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)
                
                # Aggregate the final mean dice result
                metric_d = dice_metric.aggregate().item()
                
                # Apply scheduler step on plateau
                scheduler.step(metric_d)
                
                # Aggregate the final mean dices for each label
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                
                # Reset metrics objects
                dice_metric.reset()
                dice_metric_batch.reset()
                
                # Add these metrics to tensorboard
                writer.add_scalar("Evaluation/Mean_dice", metric_d, epoch+1)
                writer.add_scalar("Evaluation/Tumor core dice", metric_tc, epoch+1)
                writer.add_scalar("Evaluation/Whole tumor dice", metric_wt, epoch+1)
                writer.add_scalar("Evaluation/Enhancing tumor dice", metric_et, epoch+1)
                
                # Save best model (based on mean dice metric)
                if metric_d > best_mean_dice:
                    best_mean_dice = metric_d
                    best_dice_et = metric_et
                    best_dice_tc = metric_tc
                    best_dice_wt = metric_wt
                    best_epoch = epoch + 1
                    torch.save(model, os.path.join(log_dir, "best_model.pth"))
                    print("\nSaved new best model")
                # Print metrics after each epoch
                print(
                    f"\nCurrent epoch: {epoch + 1}\nCurrent mean dice: {metric_d:.4f}"
                    f"\nMean dice tc: {metric_tc:.4f}"
                    f"\nMean dice wt: {metric_wt:.4f}"
                    f"\nMean dice et: {metric_et:.4f}"
                    f"\nBest mean dice: {best_mean_dice:.4f}"
                    f" at epoch: {best_epoch}"
                )
                
    # Adding hyperparameters value to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
    writer.add_hparams(config_hparam, {"hparam/Mean_dice":best_mean_dice,
                                       "hparam/Dice_ET":best_dice_et,
                                       "hparam/Dice_TC":best_dice_tc,
                                       "hparam/Dice_WT":best_dice_wt,
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
