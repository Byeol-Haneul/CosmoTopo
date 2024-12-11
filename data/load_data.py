import os
import torch
import torch.distributed as dist
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config.param_config import PARAM_STATS, PARAM_ORDER
from torch_sparse import SparseTensor
from config.machine import *

def sparsify(tensor):
    if tensor.layout == torch.sparse_coo:
        tensor = SparseTensor.from_torch_sparse_coo_tensor(tensor)
    return tensor    

def load_tensors(num_list, data_dir, label_filename, args, target_labels=None, feature_sets=None):
    label_file = pd.read_csv(label_filename, sep='\s+', header=0)
    
    tensor_dict = {feature: [] for feature in feature_sets}
    tensor_dict['y'] = []

    for target_label in target_labels:
        if target_label not in list(PARAM_STATS.keys()):
            raise Exception("Invalid Parameter, or Derived Parameter.")

    for num in tqdm(num_list):
        if TYPE == "Quijote":
            y = torch.Tensor(label_file.loc[num].to_numpy().astype(float))
        else:
            # CAMELS start with LH_{num} so trim first col
            y = torch.Tensor(label_file.loc[num].to_numpy()[1:-1].astype(float)) 

        # Now, y perfectly follows the defined PARAM_ORDER.
        if target_labels:
            indices = [PARAM_ORDER.index(label) for label in target_labels]
            y = y[indices]

        tensor_dict['y'].append(y)

        for feature in feature_sets:
            if feature == 'global_feature':
                continue

            tensor = torch.load(os.path.join(data_dir, f"{feature}_{num}.pt"))
            
            if feature[0] == 'x':
                feature_index = int(feature.split('_')[-1])
                tensor = tensor[:, :args.in_channels[feature_index]]  # Slice based on in_channels

                #If we only use positions, x_0 will be filled with random vals from uniform distribution
                if args.only_positions and feature_index == 0: 
                    tensor = torch.rand_like(tensor)

            tensor_dict[feature].append(tensor)

        # Calculate global features
        if 'global_feature' in feature_sets:
            feature_list = [tensor_dict[f"x_{i}"][-1].shape[0] for i in range(4)]
            global_feature = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(0)  # Shape [1, 4]
            global_feature = torch.log10(global_feature + 1)
            tensor_dict['global_feature'].append(global_feature)
    
    return tensor_dict


def split_data(lst, test_size=0.15, val_size=0.15):
    train, temp = train_test_split(lst, test_size=test_size + val_size, shuffle=False)
    val, test = train_test_split(temp, test_size=test_size / (test_size + val_size), shuffle=False)
    return train, val, test