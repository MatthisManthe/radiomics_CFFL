import os
import json
import torch
import numpy as np
import monai
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import DataLoader, decollate_batch, CacheDataset, LMDBDataset
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pacmap
import pandas as pd
import pickle

variable_explorer = None


def compute_GMM(config, full_flair_list, log_dir):
    
    # Load radiomics files, filter train_val samples
    all_mod_part_results = {}
    for mod in ["flair", "t1", "t1ce", "t2"]:
        with open(os.path.join(config["radiomic_dir"], config["part_file"].replace("#", mod)), "r") as file:
            part_results_load = json.load(file)
            all_mod_part_results[mod] = part_results_load
            
    results_flair = all_mod_part_results["flair"]
    
    # Fix an arbitrary order for the features in a vector
    keys = list(results_flair.values())[0][0].keys()
    
    print("All keys per modality: ", keys, len(keys), "\n")
    
    final_keys = []
    for k in keys:
        if ("shape" not in k) and ("diagnostics" not in k) and (k != "id_sample"):
            final_keys.append(k)
    
    print("Final features per modality: ", final_keys, len(final_keys))
    
    unnorm_data_matrix = []
    sample_clients = []
    sample_ids = []
    
    for (idm, (mod, part_results)) in enumerate(all_mod_part_results.items()):
        ids = 0
        for (idc, (client, v)) in enumerate(part_results.items()):
            for (idv, volume_radiomics) in enumerate(v):
                if os.path.basename(volume_radiomics["id_sample"].replace(mod, "flair")) in full_flair_list:
                    if idm == 0:
                        normalized_features = []
                        for (idx, k) in enumerate(final_keys):
                            normalized_features.append(volume_radiomics[k])
                        unnorm_data_matrix.append(normalized_features)
                        sample_clients.append(client)
                        sample_ids.append(volume_radiomics["id_sample"])
                    else:
                        normalized_features = []
                        for (idx, k) in enumerate(final_keys):
                            normalized_features.append(volume_radiomics[k])
                        unnorm_data_matrix[ids].extend(normalized_features)
                    ids += 1
                
    unnorm_data_matrix = np.array(unnorm_data_matrix)
    colors = np.array(sample_clients)
    print("Unnorm data matrix shape: ", unnorm_data_matrix.shape)
    
    # Compute normalization function
    min_scale_values_per_key = np.percentile(unnorm_data_matrix, 2, axis=0)
    max_scale_values_per_key = np.percentile(unnorm_data_matrix, 98, axis=0)
    print("Length min scale values per key: ", len(min_scale_values_per_key))
    #print(min_scale_values_per_key, max_scale_values_per_key)
    
    data_matrix = []
    for sample in unnorm_data_matrix:
        normalized_sample = []
        for (idf, feature) in enumerate(sample):
            value = feature
            value = (value - min_scale_values_per_key[idf])/(max_scale_values_per_key[idf] - min_scale_values_per_key[idf])
            value = max(value, 0.0)
            value = min(1.0, value)
            normalized_sample.append(value)
        data_matrix.append(normalized_sample)
    data_matrix = np.array(data_matrix)
    print("Shape datamatrix ", data_matrix.shape)
    
    n_components = config["pca_components"]
    pca = PCA(n_components=n_components)
    pca_data_matrix = pca.fit_transform(data_matrix)
    print("Exaplained variance per component", pca.explained_variance_ratio_)
    print("Total explained variance with", n_components, "components:", sum(pca.explained_variance_ratio_))
    print("Singular values:", pca.singular_values_)
    print("Actual components in order:", pca.components_)

    """
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=35)
    X_embedded = tsne.fit_transform(pca_data_matrix)
    print(X_embedded.shape)
    """

    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 

    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_embedded = embedding.fit_transform(pca_data_matrix, init="pca")
    
    # ---------------- Per institution -----------------------
    
    clients = np.unique(colors)
    
    markers = ["o", "1", "d"]
    colors_list = ["b", "g", "r", "orange", "brown", "c", "m", "y"]
    couple_mark_color_par_client = [[markers[i//len(colors_list)], colors_list[i%len(colors_list)]] for i in range(len(clients))]
    
    plt.figure(figsize=(9, 9))
    for (i, c) in enumerate(clients):
        plt.scatter(X_embedded[colors == c,0], X_embedded[colors == c,1], s=20, label=c, c=couple_mark_color_par_client[i][1], marker=couple_mark_color_par_client[i][0])
    plt.legend()
    #plt.title(f"PCA comp {n_components}, explained variance: {sum(pca.explained_variance_ratio_)}")
    plt.axis('tight')
    plt.savefig(os.path.join(log_dir, config["part_file"]+"GMM_train_val.svg"))
    #plt.show()
    
    # ------------------ GMM ---------------
    GMM_components = config["gmm_components"]
    GMM_covariance_type = config["gmm_covariance_type"]
    GMM = GaussianMixture(n_components=GMM_components, covariance_type=GMM_covariance_type, max_iter=100, n_init=20, random_state=config["gmm_random_state"])
    if config["on_pacmap"]:
        GMM.fit(X_embedded)
        labels_gmm = GMM.predict(X_embedded)
    else:
        GMM.fit(pca_data_matrix)
        labels_gmm = GMM.predict(pca_data_matrix)
    print("Gmm label shape: ", labels_gmm.shape)
    print("Gmm sample labels: ", labels_gmm)
    print("Gmm labels: ", np.unique(labels_gmm))
    # Create a scatter plot.
    markers = ["o", "1", "d"]
    colors_list = ["b", "g", "r", "orange", "brown", "c", "m", "y"]
    couple_mark_color_per_cluster = [[markers[i//len(colors_list)], colors_list[i%len(colors_list)]] for i in range(GMM_components)]
    
    plt.figure(figsize=(9, 9))
    for i in range(GMM_components):
        plt.scatter(X_embedded[labels_gmm == i,0], X_embedded[labels_gmm == i,1], s=20, label=i, c=couple_mark_color_per_cluster[i][1], marker=couple_mark_color_per_cluster[i][0])
    plt.legend()
    plt.title(f"PCA comp {n_components}, explained variance: {sum(pca.explained_variance_ratio_)}")
    plt.axis('tight')
    plt.savefig(os.path.join(log_dir, config["part_file"]+"GMM_train_val.svg"))
    #plt.show()
    
    np.save(os.path.join(log_dir, "final_keys.npy"), np.array(final_keys))
    np.save(os.path.join(log_dir, "min_scale_values_per_key.npy"), min_scale_values_per_key)
    np.save(os.path.join(log_dir, "max_scale_values_per_key.npy"), max_scale_values_per_key)
    pickle.dump(pca, open(os.path.join(log_dir, "PCA_model.sav"), 'wb'))
    pickle.dump(GMM, open(os.path.join(log_dir, "GMM_model.sav"), 'wb'))
    
    quit()
    
    return pca, GMM, min_scale_values_per_key, max_scale_values_per_key
    
    
    

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
    clients = list(samples_inst_map_train.keys())
    
    # Define the device to use (CPU or which GPU)
    device = torch.device(config["device"])
    
    # Define the transformations to be applied to data
    train_transform = generate_train_tranform(roi_size=config["roi_size"], data_aug=config["data_augmentation"], device=device)
    val_transform = generate_val_tranform(roi_size=config["roi_size"])
    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )
    
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
    
    pca, gmm, min_scale_values_per_key, max_scale_values_per_key = compute_GMM(config, full_flair_list, log_dir)
    
    modalities = ["flair", "t1", "t1ce", "t2"]
    
    # Load radiomics files, filter train_val samples
    all_mod_part_results = {}
    for mod in modalities:
        with open(os.path.join(config["radiomic_dir"], config["part_file"].replace("#", mod)), "r") as file:
            part_results_load = json.load(file)
            all_mod_part_results[mod] = part_results_load
            
    results_flair = all_mod_part_results["flair"]
    
    # Fix an arbitrary order for the features in a vector
    keys = list(results_flair.values())[0][0].keys()
    
    print("All keys per modality: ", keys, len(keys), "\n")
    
    final_keys = []
    for k in keys:
        if ("shape" not in k) and ("diagnostics" not in k) and (k != "id_sample"):
            final_keys.append(k)
    
    print("Final features per modality: ", final_keys, len(final_keys))
    
    # Hell, un normalized features
    dict_radiomics = {client:{} for client in clients}
    
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
                        dict_radiomics[client][basename] = features
                    else:
                        features = []
                        for (idx, k) in enumerate(final_keys):
                            features.append(volume_radiomics[k])
                        basename = os.path.basename(volume_radiomics["id_sample"].replace(mod, "flair"))
                        dict_radiomics[client][basename].extend(features)
    
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
            
    # Assign a cluster to each sample, and build train and val dict
    clustered_train_part_list = {i:{} for i in range(config["gmm_components"])}
    clustered_val_part_list = {i:{} for i in range(config["gmm_components"])}
    clustered_train = {i:[] for i in range(config["gmm_components"])}
    clustered_val = {i:[] for i in range(config["gmm_components"])}
    
    for (client, samples) in dict_radiomics.items():
        print(client)
        for (sample, features) in samples.items():
            reduced_features = pca.transform([features])
            label = gmm.predict(reduced_features)[0]
            if sample in train_flair_list:
                if not client in clustered_train_part_list[label]:
                    clustered_train_part_list[label][client] = [train_flair_list_to_dataset[sample]]
                else:
                    clustered_train_part_list[label][client].append(train_flair_list_to_dataset[sample])
                clustered_train[label].append(train_flair_list_to_dataset[sample])
            else:
                if not client in clustered_val_part_list[label]:
                    clustered_val_part_list[label][client] = [val_flair_list_to_dataset[sample]]
                else:
                    clustered_val_part_list[label][client].append(val_flair_list_to_dataset[sample])
                clustered_val[label].append(val_flair_list_to_dataset[sample])
    
    with open(os.path.join(log_dir, 'clustered_training_samples.json'), 'w') as fp:
        json.dump(clustered_train_part_list, fp, indent=4)
        
    with open(os.path.join(log_dir, 'clustered_val_samples.json'), 'w') as fp:
        json.dump(clustered_val_part_list, fp, indent=4)
      
    
    # Initialize lists of train and validation datasets and loaders
    train_ds_dict = {}
    train_loader_dict = {}
    train_length_dict = {}
    val_ds_dict = {}
    val_loader_dict = {}
    val_length_dict = {}

    for cluster in range(config["gmm_components"]):
        train_ds_dict[cluster] = CacheDataset(data=clustered_train[cluster], transform=train_transform)
        train_loader_dict[cluster] = DataLoader(train_ds_dict[cluster], batch_size=config["batch_size"], shuffle=True, num_workers=0)
        train_length_dict[cluster] = len(train_ds_dict[cluster])
    
    for cluster in range(config["gmm_components"]):
        val_ds_dict[cluster] = CacheDataset(data=clustered_val[cluster], transform=val_transform)
        val_loader_dict[cluster] = DataLoader(val_ds_dict[cluster], batch_size=config["batch_size"], shuffle=False, num_workers=0)
        val_length_dict[cluster] = len(val_ds_dict[cluster])
    
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
    
    cluster_models = {cluster:DynUNet(
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
                    ).to(device) for cluster in range(config["gmm_components"])}
    
    # Define loss function, optimizer and metrics
    loss_function = DiceLoss(sigmoid=True, smooth_nr=1, smooth_dr=1, squared_pred=False)
    
    optimizer_dict = {}
    for cluster in range(config["gmm_components"]):
        optimizer_dict[cluster] = torch.optim.SGD(cluster_models[cluster].parameters(), config["learning_rate"], weight_decay=config["weight_decay"])
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=False)
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    
    # Adding graph of model to tensorboard and print it
    #writer.add_graph(local_models[0]["11"], next(iter(train_loader_dict[0]["11"]))["image"].to(device))
    #print(summary(local_models[0]["1"], next(iter(train_loader_dict[0]["1"]))["image"].to(device), show_input=False, show_hierarchical=True))
        
    result_dict = {}
    best_mean_dices = []
    best_dices_et = []
    best_dices_tc = []
    best_dices_wt = []

    for cluster in range(config["gmm_components"]):
        
        # Initialize metrics
        best_mean_dice = -1
        best_dice_et = -1
        best_dice_tc = -1
        best_dice_wt = -1
        best_comm = 0
        
        if config["pretrain_fedavg"]:
            cluster_models[cluster].load_state_dict(torch.load(os.path.join(config["model_dir"], config["model_file"])))
        else:
            cluster_models[cluster].load_state_dict(cluster_models[0].state_dict())
        
        # Evaluation at step 0
        cluster_models[cluster].eval()
        
        # Local metrics
        with torch.no_grad(): 
            
            for (idx, val_data) in enumerate(tqdm(val_loader_dict[cluster])):
                
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                
                # Sliding window inference and thresholding
                val_outputs = sliding_window_inference(val_inputs, config["roi_size"], 1, cluster_models[cluster], overlap=config["sliding_window_overlap"], mode="gaussian")
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

        writer.add_scalar(f"Validation/Cluster {cluster} model/Mean_dice", metric_d, 0)
        writer.add_scalar(f"Validation/Cluster {cluster} model/Tumor core dice", metric_tc, 0)
        writer.add_scalar(f"Validation/Cluster {cluster} model/Whole tumor dice", metric_wt, 0)
        writer.add_scalar(f"Validation/Cluster {cluster} model/Enhancing tumor dice", metric_et, 0)

        if metric_d > best_mean_dice:
            best_mean_dice = metric_d
            best_dice_et = metric_et
            best_dice_tc = metric_tc
            best_dice_wt = metric_wt
            best_comm = 0
            torch.save(cluster_models[cluster].state_dict(), os.path.join(log_dir, f"best_cluster_{cluster}_model.pth"))
            print("\nSaved new best metric global model")
        print(
            f"\nCurrent epoch: 0, current mean dice: {metric_d:.4f}"
            f"\nmean dice tc: {metric_tc:.4f}"
            f"\nmean dice wt: {metric_wt:.4f}"
            f"\nmean dice et: {metric_et:.4f}"
            f"\nbest mean dice: {best_mean_dice:.4f} "
            f"at global epoch: {best_comm}"
        ) 
        
        # Training process
        for epoch in range(config["max_comm_rounds"]):
            
            print("-" * 10)
            print(f"epoch {epoch + 1}/{config['max_comm_rounds']}")
                
            # ----------- Local training step -------------
            cluster_models[cluster].train()
            
            local_epoch_loss = 0
            step = 0

            for (idx, batch_data) in enumerate(tqdm(train_loader_dict[cluster])):
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )

                optimizer_dict[cluster].zero_grad()
                outputs = cluster_models[cluster](inputs)
                
                # Plot predictions
                if epoch==0 and idx<4 and False:
                    with torch.no_grad():
                        fig = plot_train_image_pred_output(inputs, outputs, labels)
                        writer.add_figure(f"Initial check/Cluster {cluster}", fig, epoch+1, close=False)
                        plt.show()
                    
                # Evaluate loss, backward and step from optimizer
                loss = loss_function(outputs, labels)
                
                loss.backward()
                optimizer_dict[cluster].step()

                # Get loss value on last batch
                local_epoch_loss += loss.item()*inputs.shape[0]
                # print(f"Client {client}, step {step}/{np.ceil(len(train_ds_list[client]) / train_loader_list[client].batch_size)}, f"train_loss: {loss.item():.4f}")

            # lr_scheduler.step()
            local_epoch_loss /= train_length_dict[cluster]

            writer.add_scalar(f"Loss/train/Cluster {cluster}", local_epoch_loss, epoch+1)
            print(f"Cluster {cluster}, cluster epoch {epoch + 1}, average loss: {local_epoch_loss:.4f}")

            # Evaluation of global model
            if (epoch+1) % config["global_validation_interval"] == 0:
        
                cluster_models[cluster].eval()
                    
                with torch.no_grad(): 
                    
                    for (idx, val_data) in enumerate(tqdm(val_loader_dict[cluster])):
                        
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        
                        # Sliding window inference and thresholding
                        val_outputs = sliding_window_inference(val_inputs, config["roi_size"], 1, cluster_models[cluster], overlap=config["sliding_window_overlap"], mode="gaussian")
                        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
    
                        # Plot predictions on evaluation set
                        if (epoch + 1) % config["global_print_validation_interval"] == 0 and False:
                            fig = plot_labels_outputs(val_inputs, val_outputs, val_labels)
                            writer.add_figure(f"Validation plot/Cluster model/{epoch+1}", fig, epoch+1, close=False)
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
        
                writer.add_scalar(f"Validation/Cluster {cluster} model/Mean_dice", metric_d, epoch+1)
                writer.add_scalar(f"Validation/Cluster {cluster} model/Tumor core dice", metric_tc, epoch+1)
                writer.add_scalar(f"Validation/Cluster {cluster} model/Whole tumor dice", metric_wt, epoch+1)
                writer.add_scalar(f"Validation/Cluster {cluster} model/Enhancing tumor dice", metric_et, epoch+1)
        
                if metric_d > best_mean_dice:
                    best_mean_dice = metric_d
                    best_dice_et = metric_et
                    best_dice_tc = metric_tc
                    best_dice_wt = metric_wt
                    best_comm = epoch + 1
                    torch.save(cluster_models[cluster].state_dict(), os.path.join(log_dir, f"best_cluster_{cluster}_model.pth"))
                    print("\nSaved new best metric global model")
                print(
                    f"\nCurrent epoch: {epoch + 1}, current mean dice: {metric_d:.4f}"
                    f"\nmean dice tc: {metric_tc:.4f}"
                    f"\nmean dice wt: {metric_wt:.4f}"
                    f"\nmean dice et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_mean_dice:.4f} "
                    f"at global epoch: {best_comm}"
                ) 
        
        result_dict[f"hparam/Mean_dice/cluster_{cluster}"] = best_mean_dice
        result_dict[f"hparam/Dice_ET/cluster_{cluster}"] = best_dice_et
        result_dict[f"hparam/Dice_TC/cluster_{cluster}"] = best_dice_tc
        result_dict[f"hparam/Dice_WT/cluster_{cluster}"] = best_dice_wt
        
        best_mean_dices.append(best_mean_dice)
        best_dices_et.append(best_dice_et)
        best_dices_tc.append(best_dice_tc)
        best_dices_wt.append(best_dice_wt)
        
    num_samples = np.array([val_length_dict[cluster] for cluster in range(config["gmm_components"])], dtype=np.float64)
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
