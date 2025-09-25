import os
import numpy as np
import glob
import pandas as pd

def generate_data_dict_modality(data_dir, modality):

    images = sorted(
        glob.glob(os.path.join(data_dir, "*", "*" + modality + ".nii.gz"))
    )
    
    data_dict = [
        {"image": image_name}
        for image_name in images
    ]
    
    return data_dict


def generate_data_dict(data_dir):

    images_flair = sorted(
        glob.glob(os.path.join(data_dir, "*", "*flair.nii.gz"))
    )
    
    images_t1 = sorted(
        glob.glob(os.path.join(data_dir, "*", "*t1.nii.gz"))
    )
    
    images_t1ce = sorted(
        glob.glob(os.path.join(data_dir, "*", "*t1ce.nii.gz"))
    )
    
    images_t2 = sorted(
        glob.glob(os.path.join(data_dir, "*", "*t2.nii.gz"))
    )
    
    label = sorted(
        glob.glob(os.path.join(data_dir, "*", "*seg.nii.gz"))
    )
    
    data_dict = [
        {"image": [image_name_flair, image_name_t1, image_name_t1ce, image_name_t2], "label":label_name}
        for image_name_flair, image_name_t1, image_name_t1ce, image_name_t2, label_name in zip(images_flair, images_t1, images_t1ce, images_t2, label)
    ]
    
    return data_dict


def generate_data_dict_inference(data_dir):

    images_flair = sorted(
        glob.glob(os.path.join(data_dir, "*", "*flair.nii.gz"))
    )
    
    images_t1 = sorted(
        glob.glob(os.path.join(data_dir, "*", "*t1.nii.gz"))
    )
    
    images_t1ce = sorted(
        glob.glob(os.path.join(data_dir, "*", "*t1ce.nii.gz"))
    )
    
    images_t2 = sorted(
        glob.glob(os.path.join(data_dir, "*", "*t2.nii.gz"))
    )
    
    data_dict = [
        {"image": [image_name_flair, image_name_t1, image_name_t1ce, image_name_t2]}
        for image_name_flair, image_name_t1, image_name_t1ce, image_name_t2 in zip(images_flair, images_t1, images_t1ce, images_t2)
    ]
    
    return data_dict


def generate_data_dict_2D_modality(data_dir, modality):
    
    images = sorted(
        glob.glob(os.path.join(data_dir, "*", "slices", modality + "_*"))
    )
    
    data_dict = [
        {"image": image_name}
        for image_name in images
    ]
    
    return data_dict
    

def generate_data_dict_2D_modality_slices(data_dir, modality, slices):
    
    slices_name = ["0"+str(s) for s in range(slices[0], slices[1])]

    images = []
    for s in slices_name:
        images.extend(glob.glob(os.path.join(data_dir, "*", "slices_norm", modality + "_" + s + "*")))
    
    data_dict = [
        {"image": image_name}
        for image_name in images
    ]
    
    print(data_dict)
    
    return data_dict


def generate_train_val_dict(data_dict, prop_full_dataset=1.0, ratio_train=1.0, seed=0, min_data=True):
    
    dataset_limit = int(len(data_dict)*prop_full_dataset)
    train_set_size = int(dataset_limit*ratio_train)
    
    # Put at least one sample in training set
    if min_data and train_set_size == 0:
        train_set_size = 1
    
    # Put at least one sample in validation set
    if min_data and dataset_limit - train_set_size <= 0:
        dataset_limit = train_set_size + 1
        
    np.random.seed(seed)
    np.random.shuffle(data_dict)
    
    train_dict = data_dict[:train_set_size]
    val_dict = data_dict[train_set_size:dataset_limit]
    print("Training set size: ", len(train_dict), "Validation set size: ", len(val_dict))
    
    return train_dict, val_dict


def random_partitioning_train_val_dict(data_dict, prop_full_dataset=1.0, ratio_train=0.8, nb_clients=4, seed=0):
    
    dataset_limit = int(len(data_dict)*prop_full_dataset)
    train_set_size = int(dataset_limit*ratio_train)
    
    np.random.seed(seed)
    np.random.shuffle(data_dict)
    
    train_dict = data_dict[:train_set_size]
    val_dict = data_dict[train_set_size:dataset_limit]
    print("Training set size: ", len(train_dict), "Validation set size: ", len(val_dict))
    
    tmp_dict = train_dict.copy()
    part_train_dict = [tmp_dict[i::nb_clients] for i in range(nb_clients)]
    
    tmp_dict = val_dict.copy()
    part_val_dict = [tmp_dict[i::nb_clients] for i in range(nb_clients)]
    
    return part_train_dict, part_val_dict, train_dict, val_dict


def random_partitioning_train_val_distill_dict(data_dict, prop_full_dataset=1.0, ratio_distill=0.1, ratio_train=0.8, nb_clients=4, seed=0):
    
    dataset_limit = int(len(data_dict)*prop_full_dataset)
    distill_set_size = int(dataset_limit*ratio_distill)
    train_set_size = int((dataset_limit-distill_set_size)*ratio_train)
    
    np.random.seed(seed)
    np.random.shuffle(data_dict)
    
    distill_dict = data_dict[-distill_set_size:]
    train_dict = data_dict[:train_set_size]
    val_dict = data_dict[train_set_size:dataset_limit-distill_set_size]
    
    print("Training set size: ", len(train_dict), "Distillation set size: ", len(distill_dict), "Validation set size: ", len(val_dict))
    
    tmp_dict = train_dict.copy()
    part_train_dict = [tmp_dict[i::nb_clients] for i in range(nb_clients)]
    
    tmp_dict = val_dict.copy()
    part_val_dict = [tmp_dict[i::nb_clients] for i in range(nb_clients)]
    
    return distill_dict, part_train_dict, part_val_dict, train_dict, val_dict


# Function used to isolate around 85% of the training set into a test set
def save_csv_train_test_partitioning(ratio_train=0.85, seed=0):
    
    partition_file = "../datasets/MICCAI_FeTS2022_TrainingData/partitioning_1.csv"
    data = pd.read_csv(partition_file, dtype=str)
    print(data)
    
    train_part, test_part = pd.DataFrame(columns = data.columns), pd.DataFrame(columns = data.columns)
    samples_inst_map_train, samples_inst_map_test = {}, {}
    
    inst_names = list(data['Partition_ID'].unique())
    inst_names.sort()
    
    print(inst_names)
    
    for inst_name in inst_names:
        samples_id = data[data['Partition_ID'] == inst_name]
        print(len(samples_id))
        
        dataset_limit = len(samples_id)
        train_set_size = int(dataset_limit*ratio_train)
        print(dataset_limit, train_set_size)
        
        samples_id = samples_id.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(samples_id.iloc[0:1])
        
        train_dict = samples_id.iloc[:train_set_size]
        test_dict = samples_id.iloc[train_set_size:dataset_limit]
        print(test_dict)
        
        train_part = train_part.append(train_dict, ignore_index=True)
        test_part = test_part.append(test_dict, ignore_index=True)
    print(test_part)
    train_part.to_csv("../datasets/MICCAI_FeTS2022_TrainingData/partitioning_1_train.csv")
    test_part.to_csv("../datasets/MICCAI_FeTS2022_TrainingData/partitioning_1_test.csv")
    
    
# Function used to generate federated k folds (k folds per institution)
def save_csv_k_fold_train_test_partitioning(nb_folds=5, seed=10):
    
    partition_file = "../../datasets/MICCAI_FeTS2022_TrainingData/partitioning_1.csv"
    data = pd.read_csv(partition_file, dtype=str)
    print(data)
    
    folds_part = [pd.DataFrame(columns = data.columns) for i in range(nb_folds)]

    inst_names = list(data['Partition_ID'].unique())
    inst_names.sort()
    
    print(inst_names)
    
    np.random.seed(seed)
    too_few_samples_df = []
    
    for inst_name in inst_names:
        print("\nInstitution ", inst_name)
        samples_id = data[data['Partition_ID'] == inst_name]
        print(len(samples_id))
        
        dataset_limit = len(samples_id)
        
        if dataset_limit < nb_folds:
            samples_id = samples_id.sample(frac=1, random_state=seed).reset_index(drop=True)
            too_few_samples_df.append(samples_id)
            continue
        
        fold_sizes = np.full(nb_folds, dataset_limit // nb_folds, dtype=int)
        fold_sizes[: dataset_limit % nb_folds] += 1
        print("Before shuffle", fold_sizes)
        
        np.random.shuffle(fold_sizes)
        print("After shuffle", fold_sizes)
        
        samples_id = samples_id.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(samples_id.iloc[0:1])
        
        current = 0
        for i in range(nb_folds):
            start, stop = current, current + fold_sizes[i]
            fold = samples_id.iloc[start:stop]
            current = stop
            
            folds_part[i] = folds_part[i].append(fold, ignore_index=True)
      
    saved_test_folds = []
    
    for i in range(nb_folds):
        train_fold = pd.concat([folds_part[j] for j in range(nb_folds) if j!=i])
        print("Before complex case train shape ", train_fold.shape)
        test_fold = folds_part[i]
        print("Before complex case test shape ", test_fold.shape)
        
        # Take care of the problematic case where there are fewer samples than folds for an institution: one fold will serve as a test set multiple times
        for inst in range(len(too_few_samples_df)):
            test_index = i if i < len(too_few_samples_df[inst]) else len(too_few_samples_df[inst])-1
            test_fold = pd.concat((test_fold, too_few_samples_df[inst].iloc[test_index:test_index+1]))
            train_fold = pd.concat((train_fold, too_few_samples_df[inst].iloc[:test_index], too_few_samples_df[inst].iloc[test_index+1:]))
         
        print("After complex case train shape ", train_fold.shape)
        print("After complex case test shape ", test_fold.shape)
        print("Represented institutions in the train set ", len(train_fold["Partition_ID"].unique()))
        print("Represented institutions in the test set ", len(test_fold["Partition_ID"].unique()))
        
        print("Any duplicate? ", pd.concat((train_fold, test_fold)).duplicated().sum())
        
        saved_test_folds.append(test_fold)
        
        #train_fold.to_csv(f"../../datasets/MICCAI_FeTS2022_TrainingData/partitioning_1_folds/partitioning_1_train_fold_{i}.csv")
        #test_fold.to_csv(f"../../datasets/MICCAI_FeTS2022_TrainingData/partitioning_1_folds/partitioning_1_test_fold_{i}.csv")
    
    
    print("Any duplicate between test folds? (expected 2): ", pd.concat([test for test in saved_test_folds]).duplicated().sum())
            
    
    
    

def gen_partitioning_fets(data_dict, partition_file, prop_full_dataset=1.0, ratio_train=0.8, seed=0):
    data = pd.read_csv(partition_file, dtype=str)
    print(data)
    
    train_part_list, val_part_list, train_dict, val_dict = [], [], [], []
    samples_inst_map_train, samples_inst_map_val = {}, {}
    
    inst_names = list(data['Partition_ID'].unique())
    inst_names.sort()

    print(inst_names)

    for inst_name in inst_names:
        samples_id = data['Subject_ID'][data['Partition_ID'] == inst_name]
        print(f"\nInstitution {inst_name}, dataset size: {len(samples_id)}")

        samples_paths = []
        for sample_id in samples_id:
            sample_dict = [d_dict for d_dict in data_dict if sample_id in d_dict["image"][0] or sample_id in d_dict["image"]]
            if len(sample_dict) > 0:
                samples_paths.extend(sample_dict)
            
        train, val = generate_train_val_dict(samples_paths, prop_full_dataset=prop_full_dataset, 
                                             ratio_train=ratio_train, seed=seed)
        
        train_part_list.append(train)
        val_part_list.append(val)
        
        samples_inst_map_train[inst_name] = train.copy()
        samples_inst_map_val[inst_name] = val.copy()
        
        train_dict += train
        val_dict += val
        print("Total size of combined training and validation set: ", len(train)+len(val))
    print("\nTotal train set size: ", len(train_dict))
    print("Total val set size: ", len(val_dict))
    print(samples_inst_map_train.keys())
    
    return train_part_list, val_part_list, train_dict, val_dict, samples_inst_map_train, samples_inst_map_val


class DataLoaderWithMemory:
    """This class allows to iterate the dataloader infinitely batch by batch.
    When there are no more batches the iterator is reset silently.
    This class allows to keep the memory of the state of the iterator hence its
    name.
    """

    def __init__(self, dataloader):
        """This initialization takes a dataloader and creates an iterator object
        from it.
        Parameters
        ----------
        dataloader : torch.utils.data.dataloader
            A dataloader object built from one of the datasets of this repository.
        """
        self._dataloader = dataloader

        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader.dataset)

    def get_samples(self):
        """This method generates the next batch from the iterator or resets it
        if needed. It can be called an infinite amount of times.
        Returns
        -------
        tuple
            a batch from the iterator
        """
        try:
            X = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            X = next(self._iterator)
        return X