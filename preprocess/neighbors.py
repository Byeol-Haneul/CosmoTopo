'''
Author: Jun-Young Lee

Summary:
Generates neighborhoods from pickled combinatorial complexes, and save to torch-friendly data structure for training.

Notes:
- Node, edge, tetrahedral, cluster, and hyperedge features / neighborhoods.
- Intra-rank adjacency matrices (with self-loops).
- Inter-rank incidence matrices.
- Global graph-level summary features (log-scaled size of each rank).

References
----------
.. [TopoModelX] https://github.com/pyt-team/TopoModelX/blob/main/topomodelx/nn/combinatorial/
'''

from mpi4py import MPI
import torch
import numpy as np
from toponetx.readwrite.serialization import load_from_pickle
import pandas as pd
import os
import sys
from config_preprocess import *
import scipy 


def scipy_to_torch_sparse(scipy_mat):
    # If input is dense numpy array, convert to sparse COO first
    if isinstance(scipy_mat, np.ndarray):
        scipy_mat = scipy.sparse.coo_matrix(scipy_mat)
    else:
        scipy_mat = scipy_mat.tocoo()
    indices = torch.tensor(np.vstack((scipy_mat.row, scipy_mat.col)), dtype=torch.long)
    values = torch.tensor(scipy_mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, scipy_mat.shape)


def get_neighbors(num, cc):
    in_channels = [-1, -1, -1, -1, -1]
    results = {}

    in_filename = f"data_{num}.pickle"
    print(f"[LOG] Loading pickle file {in_filename}", file=sys.stderr)

    if cc is None:
        cc = load_from_pickle(cc_dir + in_filename)

    '''
    Features
    '''
    print(f"[LOG] Processing node features for num {num}", file=sys.stderr)
    x_0 = list(cc.get_node_attributes("node_feat").values())
    in_channels[0] = len(x_0[0])
    results['x_0'] = torch.tensor(np.stack(x_0)).reshape(-1, in_channels[0])

    print(f"[LOG] Processing edge features for num {num}", file=sys.stderr)
    x_1 = list(cc.get_cell_attributes("edge_feat").values())
    in_channels[1] = len(x_1[0])
    results['x_1'] = torch.tensor(np.stack(x_1)).reshape(-1, in_channels[1])

    print(f"[LOG] Processing tetra features for num {num}", file=sys.stderr)
    x_2 = list(cc.get_cell_attributes("tetra_feat").values())
    in_channels[2] = len(x_2[0])
    results['x_2'] = torch.tensor(np.stack(x_2)).reshape(-1, in_channels[2])

    print(f"[LOG] Processing cluster features for num {num}", file=sys.stderr)
    x_3 = list(cc.get_cell_attributes("cluster_feat").values())
    in_channels[3] = len(x_3[0])
    results['x_3'] = torch.tensor(np.stack(x_3)).reshape(-1, in_channels[3])

    print(f"[LOG] Processing cluster features for num {num}", file=sys.stderr)
    x_4 = list(cc.get_cell_attributes("hyperedge_feat").values())
    in_channels[4] = len(x_4[0])
    results['x_4'] = torch.tensor(np.stack(x_4)).reshape(-1, in_channels[4])

    '''
    Adjacency
    '''
    print(f"[LOG] Computing n0_to_0 for num {num}", file=sys.stderr)
    n0_to_0 = cc.adjacency_matrix(rank=0, via_rank=1)
    n0_to_0 += scipy.sparse.eye(n0_to_0.shape[0])
    results['n0_to_0'] = scipy_to_torch_sparse(n0_to_0)

    print(f"[LOG] Computing n1_to_1 for num {num}", file=sys.stderr)
    n1_to_1 = cc.adjacency_matrix(rank=1, via_rank=2)
    n1_to_1 += scipy.sparse.eye(n1_to_1.shape[0])
    results['n1_to_1'] = scipy_to_torch_sparse(n1_to_1)

    if FLAG_HIGHER_ORDER:
        print(f"[LOG] Computing n2_to_2 (adjacency)", file=sys.stderr)
        n2_to_2 = cc.adjacency_matrix(rank=2, via_rank=3)
        n2_to_2 += scipy.sparse.eye(n2_to_2.shape[0])
        results['n2_to_2'] = scipy_to_torch_sparse(n2_to_2)

        print(f"[LOG] Computing n3_to_3 (adjacency)", file=sys.stderr)
        n3_to_3 = cc.adjacency_matrix(rank=3, via_rank=4)
        n3_to_3 += scipy.sparse.eye(n3_to_3.shape[0])
        results['n3_to_3'] = scipy_to_torch_sparse(n3_to_3)

        print(f"[LOG] Computing n4_to_4 (coadjacency)", file=sys.stderr)
        n4_to_4 = cc.coadjacency_matrix(rank=4, via_rank=3)
        n4_to_4 += scipy.sparse.eye(n4_to_4.shape[0])
        results['n4_to_4'] = scipy_to_torch_sparse(n4_to_4)


    '''
    Incidence
    '''
    print(f"[LOG] Computing n0_to_1", file=sys.stderr)
    results['n0_to_1'] = scipy_to_torch_sparse(cc.incidence_matrix(rank=0, to_rank=1))

    if FLAG_HIGHER_ORDER:
        for (i, j) in [
            (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 4),
            (2, 3), (2, 4)
        ]:
            print(f"[LOG] Computing n{i}_to_{j}", file=sys.stderr)
            mat = cc.incidence_matrix(rank=i, to_rank=j)
            results[f'n{i}_to_{j}'] = scipy_to_torch_sparse(mat)

        print(f"[LOG] Computing n3_to_4", file=sys.stderr)
        try:
            n3_to_4 = cc.incidence_matrix(rank=3, to_rank=4)
            results['n3_to_4'] = scipy_to_torch_sparse(n3_to_4)
        except Exception as e:
            print(f"[ERROR] n3_to_4: {e}", file=sys.stderr)

    else:
        for i in [1, 2, 3, 4]:
            for j in [2, 3, 4]:
                results[f'n{i}_to_{j}'] = None

    '''
    Global Feature
    '''
    print(f"[LOG] Global feature for num {num}", file=sys.stderr)
    feature_list = [results[f'x_{i}'].shape[0] for i in range(5)]
    global_feature = torch.tensor(feature_list, dtype=torch.float32).unsqueeze(0)
    results['global_feature'] = torch.log10(global_feature + 1)

    '''
    Save
    '''
    print(f"[LOG] Saving tensor", file=sys.stderr)
    torch.save(results, os.path.join(tensor_dir, f"sim_{num}.pt"))
    return results


def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"[LOG] MPI initialized with rank {rank} and size {size}", file=sys.stderr)

    total_nums = CATALOG_SIZE 
    nums_per_rank = total_nums // size
    start_num = rank * nums_per_rank
    end_num = start_num + nums_per_rank

    if rank == size - 1:
        end_num = total_nums

    print(f"[LOG] Rank {rank} processing range {start_num} to {end_num}", file=sys.stderr)

    for num in range(start_num, end_num):
        print(f"[LOG] Rank {rank} processing num {num}", file=sys.stderr)
        get_neighbors(num, None)

    print(f"[LOG] Rank {rank} completed processing", file=sys.stderr)

if __name__ == "__main__":
    main()
