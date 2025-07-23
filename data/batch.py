'''
Author: Jun-Young Lee

This module provides batching utilities for higher-order cells and their neighborhood matrices,
represented using `torch_sparse.SparseTensor`. It includes efficient and flexible
support for block-diagonal batching of sparse matrices (including non-square),
along with collating dense features, global features, and target variables.

Functions:
    - block_diag_sparse: Construct a block-diagonal sparse matrix from a list of SparseTensors.
    - collate_topological_batch: Collate a list of sample dictionaries into a batched format 
      suitable for input into neural networks.

'''

import torch
from torch_sparse import SparseTensor

def block_diag_sparse(mats):
    """
    Vectorized block diagonal batching for sparse tensors (torch_sparse.SparseTensor),
    supporting non-square matrices.
    """
    # Ensure all mats are coalesced
    #mats = [mat.coalesce() for mat in mats]
    
    row_sizes = torch.tensor([mat.size(0) for mat in mats])
    col_sizes = torch.tensor([mat.size(1) for mat in mats])
    
    row_offsets = torch.cumsum(row_sizes, dim=0) - row_sizes
    col_offsets = torch.cumsum(col_sizes, dim=0) - col_sizes
    
    rows, cols, vals = [], [], []
    
    for mat, roff, coff in zip(mats, row_offsets, col_offsets):
        row, col, val = mat.coo()
        rows.append(row + roff)
        cols.append(col + coff)
        vals.append(val)
    
    row_all = torch.cat(rows)
    col_all = torch.cat(cols)
    val_all = torch.cat(vals)
    
    total_rows = row_sizes.sum().item()
    total_cols = col_sizes.sum().item()
    
    return SparseTensor(row=row_all, col=col_all, value=val_all, sparse_sizes=(total_rows, total_cols))


def collate_topological_batch(batch):
    x_keys = [k for k in batch[0].keys() if k.startswith('x_')]
    sparse_keys = [k for k in batch[0].keys() if k.startswith('n') or k.startswith('cci')]
    
    out = {}

    # Concatenate dense feature tensors
    for k in x_keys:
        out[k] = torch.cat([d[k] for d in batch], dim=0)

    # Block-diagonal sparse matrices
    for k in sparse_keys:
        mats = [d[k] for d in batch]
        out[k] = block_diag_sparse(mats)

    # Batch assignment vectors (e.g. batch_0, batch_1, ...)
    for k in sorted(x_keys):
        batch_vec = torch.cat([
            torch.full((d[k].shape[0],), idx, dtype=torch.long)
            for idx, d in enumerate(batch)
        ])
        out[f'batch_{k[2:]}'] = batch_vec  # strip 'x_' → get rank

    if 'global_feature' in batch[0]:
        out['global_feature'] = torch.cat([d['global_feature'] for d in batch], dim=0)

    if 'y' in batch[0]:
        out['y'] = torch.stack([d['y'] for d in batch], dim=0)

    return out

