'''
Author: Jun-Young Lee

Summary:
Random cell/neighborhood-dropping for data augmentation.

Notes:
- drop_prob is given as a hyperparameter.
- cell/neighborhood-dropping available for all ranks.
'''

import torch, sys
from torch_sparse import SparseTensor

def sparsify(tensor):
    if tensor.layout == torch.sparse_coo:
        try:
            tensor = SparseTensor.from_torch_sparse_coo_tensor(tensor)
        except:
            print(f"TENSOR: {tensor.shape}", tensor, file=sys.stderr)
            raise Exception
    return tensor    

def generate_mask(sparse_tensor, drop_prob=0.1):
    """
    Generate a sparse mask tensor for the given sparse tensor based on the drop probability.

    Args:
    - sparse_tensor (torch.sparse_coo_tensor): Sparse tensor to generate a mask for.
    - drop_prob (float): Probability of dropping a connection.
    
    Returns:
    - mask (torch.sparse_coo_tensor): Sparse tensor mask indicating which values to keep.
    """
    sparse_tensor = sparse_tensor.coalesce() 
    values = sparse_tensor.values()
    mask_values = torch.rand(values.shape, dtype=torch.float) > drop_prob
    mask_indices = sparse_tensor.indices()
    mask = torch.sparse_coo_tensor(mask_indices, mask_values, sparse_tensor.size()).coalesce()
    return mask


def augment_matrix(matrix, drop_prob, mask=None):
    if drop_prob > 0.0:
        if mask is None:
            mask = generate_mask(matrix, drop_prob)
        try:
            return matrix.sparse_mask(mask)
        except:
            return matrix
    else:
        return matrix

def augment_data(data, drop_prob=0.1, cci_mode='euclidean'):
    """
    Augments a dictionary of neighborhood matrices and cci matrices by making some nonzero elements zero.

    Args:
    - data (dict): Dictionary where keys are matrix names and values are sparse COO matrices.
    - drop_prob (float): Probability of dropping a connection.
    
    Returns:
    - A new dictionary with augmented neighborhood and cci matrices.
    """
    neighborhood_keys = [f'n{i}_to_{j}' for i in range(5) for j in range(i, 5)]

    if cci_mode != 'None':
        cci_keys = [f'{cci_mode}_{i}_to_{j}' for i in range(5) for j in range(i, 5)]
        key_mapping = dict(zip(neighborhood_keys, cci_keys))
    else:
        key_mapping = {}

    augmented_dict = {}

    for neigh_key in neighborhood_keys:
        if neigh_key in data:
            if data[neigh_key] == None:
                augmented_dict[neigh_key] = None
                augmented_dict[key_mapping.get(neigh_key)] = None
                continue
                
            mask = generate_mask(data[neigh_key], drop_prob)
            augmented_dict[neigh_key] = sparsify(augment_matrix(data[neigh_key], drop_prob, mask))

            if cci_mode != 'None':
                cci_key = key_mapping.get(neigh_key)
                if cci_key in data:
                    augmented_dict[cci_key] = sparsify(augment_matrix(data[cci_key], drop_prob, mask))

        elif cci_mode != 'None':
            cci_key = key_mapping.get(neigh_key)
            if cci_key in data:
                augmented_dict[cci_key] = data[cci_key]

    for key in data:
        if key not in neighborhood_keys and key not in key_mapping.values():
            augmented_dict[key] = data[key]

    return augmented_dict

