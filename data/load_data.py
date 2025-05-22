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

def load_tensors(num_list, data_dir, label_filename, args, target_labels=None, feature_sets=None):
    
    tensor_dict = {feature: [] for feature in feature_sets}
    tensor_dict['y'] = []

    for target_label in target_labels:
        if target_label not in list(PARAM_STATS.keys()):
            raise Exception("Invalid Parameter, or Derived Parameter.")

    for num in tqdm(num_list):  
        if BENCHMARK:
            with h5py.File(HDF5_DATA_FILE, "r") as f:
                y = [f["params"][param][num] for param in target_labels]
                y = torch.Tensor(y)
        else:         
            label_file = pd.read_csv(label_filename, sep='\s+', header=0)
            if TYPE == "CAMELS" or TYPE == "CAMELS_50" or TYPE == "CAMELS_SB28" or TYPE == "fR":
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
            total_invariants = torch.load(os.path.join(data_dir, f"invariant_{num}.pt"))
        except:
            print(f"NUM: {num} is yet prepared, cannot open file")
            continue

        for feature in feature_sets:
            if feature == 'global_feature':
                continue

            # Newly Added to Create Less Files
            try:
                tensor = total_tensors[feature]  # Attempt to load from total_tensors
            except KeyError:
                try:
                    tensor = total_invariants[feature]  # Fall back to total_invariants
                except KeyError:
                    tensor = None
                    #raise KeyError(f"Feature '{feature}' not found in either total_tensors or total_invariants.")

            if feature[0] == 'x':
                feature_index = int(feature.split('_')[-1])
                tensor = tensor[:, :args.in_channels[feature_index]]  # Slice based on in_channels

                #If we only use positions, x_0 will be filled with random vals from uniform distribution
                #This might be misleading. We can use only_positions=False if we use raw positions (X,Y,Z)
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