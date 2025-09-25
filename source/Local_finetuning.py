import os
import json
import torch
import numpy as np
import monai
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, Dataset, LMDBDataset
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
#from pytorch_model_summary import summary
from time import gmtime, strftime
import argparse
import shutil
import copy

from data.data_utils import generate_data_dict, gen_partitioning_fets
from data.data_preprocessing import generate_train_tranform, generate_val_tranform
from plot_utils import plot_train_image_pred_output, plot_labels_outputs
from models.models import modified_get_conv_layer

variable_explorer = None

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
    
    # Define the number of clients in the federation
    nb_clients = len(train_part_list)
    
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    
    # Define the transformations to be applied to data
    train_transform = generate_train_tranform(roi_size=config["roi_size"], data_aug=config["data_augmentation"], device=device)
    val_transform = generate_val_tranform(roi_size=config["roi_size"])
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )

    # Initialize lists of train and validation datasets and loaders
    train_ds_list = []
    train_loader_list = []
    val_ds_list = []
    val_loader_list = []
    
    # Define train and validation datasets and dataloaders for each
    if config["persistent_dataset"]:
        # Delete every cached elements
        cache_dir = config["cache_dir"]
        filelist = glob.glob(os.path.join(cache_dir, "*"))
        for f in filelist:
            os.remove(f)
            
        for i in range(nb_clients):
            train_ds_list.append(LMDBDataset(data=train_part_list[i], transform=train_transform, cache_dir=config["cache_dir"]))
            val_ds_list.append(LMDBDataset(data=val_part_list[i], transform=val_transform, cache_dir=config["cache_dir"]))
    else:
        for i in range(nb_clients):
            train_ds_list.append(CacheDataset(data=train_part_list[i], transform=train_transform))
            val_ds_list.append(CacheDataset(data=val_part_list[i], transform=val_transform))
    
    for i in range(nb_clients):
        train_loader_list.append(DataLoader(train_ds_list[i], batch_size=config["batch_size"], shuffle=True, num_workers=0))
        val_loader_list.append(DataLoader(val_ds_list[i], batch_size=config["batch_size"], shuffle=False, num_workers=0))
    
    # Define the local models, loss functions, optimizers and metrics
    """
    local_models = [UNet(
                        spatial_dims = 3,
                        in_channels = 4,
                        out_channels = 3,
                        channels = (16, 32, 64, 128, 256),
                        strides = (2, 2, 2, 2),
                        act=("LeakyReLu", {"negative_slope":0.01}),
                    ).to(device) for i in range(nb_clients)]"""
    monai.networks.blocks.dynunet_block.get_conv_layer = modified_get_conv_layer
    """
    local_models = [DynUNet(
        spatial_dims = 3,
        in_channels = 4,
        out_channels = 3,
        kernel_size = (3, 3, 3, 3),
        filters = (16, 32, 64, 128),
        strides = (1, 2, 2, 2),
        upsample_kernel_size = (2, 2, 2),
        norm_name=("INSTANCE", {"affine": False}),
        act_name=("LeakyReLu", {"negative_slope":0.01}),
        trans_bias=True,
    ).to(device) for i in range(nb_clients)]
    """
    local_models = [torch.load(os.path.join(config["model_dir"], config["model_file"])) for i in range(nb_clients)]
    
    # Define loss function, optimizer and metrics
    loss_function = DiceLoss(sigmoid=True, smooth_nr=1, smooth_dr=1, squared_pred=False)
    
    optimizer_list = None
    if config["optim"] == "sgd":
        optimizer_list = [torch.optim.SGD(local_models[i].parameters(), config["learning_rate"], weight_decay=config["weight_decay"]) for i in range(nb_clients)]
    else:
        optimizer_list = [torch.optim.Adam(local_models[i].parameters(), config["learning_rate"], weight_decay=config["weight_decay"]) for i in range(nb_clients)]
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    # Adding graph of model to tensorboard and print it
    writer.add_graph(local_models[0], next(iter(train_loader_list[0]))["image"].to(device))
    #print(summary(local_models[0], next(iter(train_loader_list[0]))["image"].to(device), show_input=False, show_hierarchical=True))
        
    result_dict = {}
    best_mean_dices = []
    best_dices_et = []
    best_dices_tc = []
    best_dices_wt = []

    # --------------------- First simulate each client sequentially ---------------------
    for client in range(nb_clients):
        
        id_client = list(samples_inst_map_train.keys())[client]
        
        # Initialize metrics
        best_mean_dice = -1
        best_dice_et = -1
        best_dice_tc = -1
        best_dice_wt = -1
        best_local_epoch = 0
        
        # Initialize the local model as the best obtained federated model.
        #local_models[client].load_state_dict(torch.load(os.path.join(config["model_dir"], config["model_file"])))

        # get validation result from the model without finetuning
        local_models[client].eval()
        
        with torch.no_grad():
            
            for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                
                # Put data batch in GPU
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                
                # Sliding window inference keeping the same roi size as in training
                val_outputs = sliding_window_inference(val_inputs, config["roi_size"], 1, local_models[client], overlap=config["sliding_window_overlap"], mode="gaussian")
                
                # Thresholding at 0.5 to get binary segmentation map
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric_d = dice_metric.aggregate().item()

            # Aggregate the final mean dices for each label
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_wt = metric_batch[1].item()
            metric_et = metric_batch[2].item()

            # Reset metrics objects
            dice_metric.reset()
            dice_metric_batch.reset()
            
            writer.add_scalar(f"Validation/Client {id_client}/Mean_dice", metric_d, 0)
            writer.add_scalar(f"Validation/Client {id_client}/Tumor core dice", metric_tc, 0)
            writer.add_scalar(f"Validation/Client {id_client}/Whole tumor dice", metric_wt, 0)
            writer.add_scalar(f"Validation/Client {id_client}/Enhancing tumor dice", metric_et, 0)

            if metric_d > best_mean_dice:
                best_mean_dice = metric_d
                best_dice_et = metric_et
                best_dice_tc = metric_tc
                best_dice_wt = metric_wt
                best_local_epoch = 0
                torch.save(local_models[client].state_dict(), os.path.join(log_dir, f"best_global_model_client_{id_client}.pth"))
                print(f"\nSaved new best metric local model for client {id_client}")
            print(
                f"\nCurrent epoch: 0, current mean dice: {metric_d:.4f}"
                f"\nmean dice tc: {metric_tc:.4f}"
                f"\nmean dice wt: {metric_wt:.4f}"
                f"\nmean dice et: {metric_et:.4f}"
                f"\nbest mean dice: {best_mean_dice:.4f} "
                f"at global epoch: {best_local_epoch}"
            )
        
        # Finetune the model for some epochs
        for local_epoch in range(config["max_local_epochs"]):
                
            # ----------- Local training step -------------
            
            local_models[client].train()
            
            local_epoch_loss = 0
            step = 0
    
            for (idx, batch_data) in enumerate(tqdm(train_loader_list[client])):
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
    
                optimizer_list[client].zero_grad()
                outputs = local_models[client](inputs)
                
                # Plot predictions
                if local_epoch==0 and idx<4 and False:
                    with torch.no_grad():
                        fig = plot_train_image_pred_output(inputs, outputs, labels)
                        writer.add_figure(f"Initial check/Client {id_client}", fig, local_epoch+1, close=False)
                        plt.show()
                    
                # Evaluate loss, backward and step from optimizer
                loss = loss_function(outputs, labels)
                
                loss.backward()
                optimizer_list[client].step()
    
                # Get loss value on last batch
                local_epoch_loss += loss.item()*inputs.shape[0]
                # print(f"Client {client}, step {step}/{np.ceil(len(train_ds_list[client]) / train_loader_list[client].batch_size)}, f"train_loss: {loss.item():.4f}")
    
            # lr_scheduler.step()
            local_epoch_loss /= len(train_ds_list[client])
    
            writer.add_scalar(f"Loss/train/Client {id_client}", local_epoch_loss, local_epoch+1)
            print(f"Client {client} local epoch {local_epoch + 1} average loss: {local_epoch_loss:.4f}")
            
            # --------------- Local validation step ---------------
            if (local_epoch) % config['local_validation_interval'] == 0:
                
                local_models[client].eval()
                
                with torch.no_grad():
                    
                    for (idx, val_data) in enumerate(tqdm(val_loader_list[client])):
                        
                        # Put data batch in GPU
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        
                        # Sliding window inference keeping the same roi size as in training
                        val_outputs = sliding_window_inference(val_inputs, config["roi_size"], 1, local_models[client], overlap=config["sliding_window_overlap"], mode="gaussian")
                        
                        # Thresholding at 0.5 to get binary segmentation map
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
    
                        # Show some images if needed
                        if (local_epoch + 1) % config['local_print_validation_interval'] == 0 and False:
                            fig = plot_labels_outputs(val_inputs, val_outputs, val_labels)
                            writer.add_figure(f"Validation plot/Client {id_client}/{local_epoch+1}", fig, local_epoch+1, close=False)
                            plt.show()
    
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)
                        dice_metric_batch(y_pred=val_outputs, y=val_labels)
    
                    # aggregate the final mean dice result
                    metric_d = dice_metric.aggregate().item()
    
                    # Aggregate the final mean dices for each label
                    metric_batch = dice_metric_batch.aggregate()
                    metric_tc = metric_batch[0].item()
                    metric_wt = metric_batch[1].item()
                    metric_et = metric_batch[2].item()
    
                    # Reset metrics objects
                    dice_metric.reset()
                    dice_metric_batch.reset()
                    
                    writer.add_scalar(f"Validation/Client {id_client}/Mean_dice", metric_d, local_epoch+1)
                    writer.add_scalar(f"Validation/Client {id_client}/Tumor core dice", metric_tc, local_epoch+1)
                    writer.add_scalar(f"Validation/Client {id_client}/Whole tumor dice", metric_wt, local_epoch+1)
                    writer.add_scalar(f"Validation/Client {id_client}/Enhancing tumor dice", metric_et, local_epoch+1)
        
                    if metric_d > best_mean_dice:
                        best_mean_dice = metric_d
                        best_dice_et = metric_et
                        best_dice_tc = metric_tc
                        best_dice_wt = metric_wt
                        best_local_epoch = local_epoch + 1
                        torch.save(local_models[client].state_dict(), os.path.join(log_dir, f"best_global_model_client_{id_client}.pth"))
                        print(f"\nSaved new best metric local model for client {id_client}")
                    print(
                        f"\nCurrent epoch: {local_epoch + 1}, current mean dice: {metric_d:.4f}"
                        f"\nmean dice tc: {metric_tc:.4f}"
                        f"\nmean dice wt: {metric_wt:.4f}"
                        f"\nmean dice et: {metric_et:.4f}"
                        f"\nbest mean dice: {best_mean_dice:.4f} "
                        f"at global epoch: {best_local_epoch}"
                    )
            
        result_dict[f"hparam/Mean_dice/client_{id_client}"] = best_mean_dice
        result_dict[f"hparam/Dice_ET/client_{id_client}"] = best_dice_et
        result_dict[f"hparam/Dice_TC/client_{id_client}"] = best_dice_tc
        result_dict[f"hparam/Dice_WT/client_{id_client}"] = best_dice_wt
        
        best_mean_dices.append(best_mean_dice)
        best_dices_et.append(best_dice_et)
        best_dices_tc.append(best_dice_tc)
        best_dices_wt.append(best_dice_wt)
        
    num_samples = np.array([len(val_ds_list[i]) for i in range(nb_clients)], dtype=np.float64)
    num_samples /= np.sum(num_samples)
    
    result_dict["hparam/Mean_dice"] = np.dot(best_mean_dices, num_samples)
    result_dict["hparam/Dice_ET"] = np.dot(best_dices_et, num_samples)
    result_dict["hparam/Dice_TC"] = np.dot(best_dices_tc, num_samples)
    result_dict["hparam/Dice_WT"] = np.dot(best_dices_wt, num_samples)
    
    # Adding hyperparameters value to tensorboard
    config_hparam = {}
    for key, value in config.items():
        if type(value) is list:
            value = torch.Tensor(value)
        config_hparam[key] = value
    writer.add_hparams(config_hparam, result_dict)    
                    


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
