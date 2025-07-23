'''
Author: Jun-Young Lee

Summary:
This module handles loading and preprocessing of simulation tensor data, to be fed to the neural networks.

Main Functions:
- load_tensors: 
    Loads features and target parameters for a list of simulation IDs..

- split_data: 
    Splits a list of sample indices into train/val/test splits using deterministic, non-shuffled partitioning, 
    for a fair comparison across different models. Sanity check on the uniformity of sampled parameters complete. 

- sparsify: 
    Converts sparse COO tensors into SparseTensor for efficient calculation.

- used_keys:
    Determines the tensors to be loaded per TNN architecture, to prevent overutilization of GPU memory.
'''

import os
import torch
import torch.distributed as dist
import pandas as pd
import h5py 

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config.param_config import PARAM_STATS, PARAM_ORDER
from torch_sparse import SparseTensor
from config.machine import *

def sparsify(tensor):
    if tensor.layout == torch.sparse_coo:
        tensor = SparseTensor.from_torch_sparse_coo_tensor(tensor)
    return tensor    

def generate_keys(x, y, cci_mode):
    keys = [f"n{x}_to_{y}"]
    if cci_mode != "None" and cci_mode:
        keys.append(f"{cci_mode}{x}_to_{y}")
    return keys

def used_keys(args):
    layerType = args.layerType
    cci_mode = args.cci_mode or "None"

    features = []

    if  layerType == "GNN":
        features += generate_keys(0, 0, cci_mode)
        features += generate_keys(0, 1, cci_mode)
        features += generate_keys(1, 1, cci_mode)
        features += ['x_0', 'x_1']

    elif layerType == "TetraTNN":
        features += ['x_0', 'x_1', 'x_2']
        features += generate_keys(0, 0, cci_mode)
        features += generate_keys(1, 1, cci_mode)
        features += generate_keys(2, 2, cci_mode)
        features += generate_keys(0, 1, cci_mode)
        features += generate_keys(1, 2, cci_mode)

    elif layerType == "ClusterTNN":
        features += ['x_0', 'x_1', 'x_3']
        features += generate_keys(0, 0, cci_mode)
        features += generate_keys(1, 1, cci_mode)
        features += generate_keys(3, 3, cci_mode)
        features += generate_keys(0, 1, cci_mode)
        features += generate_keys(0, 3, cci_mode)

    elif layerType == "TNN":
        features += ['x_0', 'x_1', 'x_2', 'x_3', 'x_4']
        for i in range(5):
            features += generate_keys(i, i, cci_mode)
        for src in range(4):
            for tgt in range(src + 1, 5):
                features += generate_keys(src, tgt, cci_mode)

    else:
        raise ValueError(f"Unknown layerType: {layerType}")

    features.append('global_feature')
    features.append('y')
    return sorted(set(features))

def load_tensors(num_list, data_dir, label_filename, args, target_labels=None):
    feature_sets = used_keys(args)
    tensor_dict = {key: [] for key in feature_sets}

    for target_label in target_labels:
        if target_label not in list(PARAM_STATS.keys()):
            raise Exception("Invalid Parameter, or Derived Parameter.")

    for num in tqdm(num_list):  
        label_file = pd.read_csv(label_filename, sep='\s+', header=0)
        if TYPE == "CAMELS":
            y = torch.Tensor(label_file.loc[num].to_numpy()[1:].astype(float)) # CAMELS and Quijote-MG start with LH_{num}/{num} so trim first col
        else:
            y = torch.Tensor(label_file.loc[num].to_numpy().astype(float))

        # Now, y perfectly follows the defined PARAM_ORDER.
        if target_labels:
            indices = [PARAM_ORDER.index(label) for label in target_labels]
            y = y[indices]

        tensor_dict['y'].append(y)

        # Newly Added to Create Less Files
        try:
            total_tensors = torch.load(os.path.join(data_dir, f"sim_{num}.pt"))
            if args.cci_mode != "None":
                total_invariants = torch.load(os.path.join(data_dir, f"invariant_{num}.pt"))
            else:
                total_invariants = None
        except:
            print(f"NUM: {num} is yet prepared, cannot open file")
            continue

        for feature in feature_sets:
            if feature == 'y' or feature == 'global_feature':
                continue

            # Newly Added to Create Less Files
            try:
                tensor = total_tensors[feature]  # Attempt to load from total_tensors
            except KeyError:
                try:
                    if total_invariants == None:
                        tensor = None
                    else:
                        tensor = total_invariants[feature]  # Fall back to total_invariants
                except KeyError:
                    tensor = None
                    #raise KeyError(f"Feature '{feature}' not found in either total_tensors or total_invariants.")

            if feature[0] == 'x':
                feature_index = int(feature.split('_')[-1])
                tensor = tensor[:, :args.in_channels[feature_index]].float()
                #If we only use positions, x_0 will be filled with random vals from uniform distribution
                if args.only_positions and feature_index == 0: 
                    tensor = torch.rand_like(tensor)

            tensor_dict[feature].append(tensor)

        # Calculate global features
        if 'global_feature' in feature_sets:
            feature_list = [total_tensors[f"x_{i}"].shape[0] for i in range(4)]
            global_feature = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(0)  # Shape [1, 4]
            global_feature = torch.log10(global_feature + 1)
            tensor_dict['global_feature'].append(global_feature)
    
    return tensor_dict


def split_data(lst, test_size=0.15, val_size=0.15):
    train, temp = train_test_split(lst, test_size=test_size + val_size, shuffle=False)
    val, test = train_test_split(temp, test_size=test_size / (test_size + val_size), shuffle=False)
    return train, val, test

def split_indices(indices, rank, world_size):
    base_size = len(indices) // world_size
    remainder = len(indices) % world_size
    if rank < remainder:
        start_idx = rank * (base_size + 1)
        end_idx = start_idx + base_size + 1
    else:
        start_idx = remainder * (base_size + 1) + (rank - remainder) * base_size
        end_idx = start_idx + base_size

    return indices[start_idx:end_idx]