import logging
import os, sys
import random
import pandas as pd
import numpy as np
import socket 
from argparse import Namespace

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from data.load_data import load_tensors, split_data
from data.dataset import CustomDataset

from model.network import Network
from train import train, evaluate, save_checkpoint, load_checkpoint
from utils.loss_functions import *

from config.param_config import PARAM_STATS, PARAM_ORDER, normalize_params, denormalize_params
from config.machine import *

from main import *

args = Namespace(
    tuning=False,
    only_positions=True,
    
    # Directories
    data_dir=DATA_DIR+"/tensors_dense/",
    model_dir="/mnt/home/jlee2/ceph/benchmark/Subset_Quijote_BSQ_rockstar_10_top5000/results/BENCHMARK_isolated_Subset_Quijote_BSQ_rockstar_10_top5000_GNN/trial_93/",
    checkpoint_dir=RESULT_DIR+"/zero_shot/",
    label_filename=LABEL_FILENAME,
    
    # Model Architecture
    in_channels=[3, 3, 5, 7, 3],
    hidden_dim=128,
    num_layers=6,
    layerType="GNN",
    attention_flag=False,
    residual_flag=True,
    
    # Target Labels
    target_labels=["Omega_m", "sigma_8"],
    
    # Training Hyperparameters
    num_epochs=300,
    test_interval=100,
    loss_fn_name="mse",
    
    T_max=64,
    learning_rate=0.0006335744983516634,
    weight_decay=1.1699938795572962e-06,
    batch_size=2,
    drop_prob=0.09919276088476181,
    update_func="tanh",
    aggr_func="all",
    
    # Device
    #device_num="0,1",
    device=None,
    
    # Dataset Split and Random Seed
    val_size=0.1,
    test_size=0.1,
    random_seed=1234,
    
    # Features & Neighborhood Functions
    cci_mode="hausdorff",
    
    feature_sets=[
    'x_0', 'x_1', 'x_2', 'x_3', 'x_4',
    'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
    'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
    'n1_to_2', 'n1_to_3', 'n1_to_4',
    'n2_to_3', 'n2_to_4',
    'n3_to_4',
    'global_feature'
    ]
)


if args.cci_mode != 'None':
    cci_list = [f'{args.cci_mode}_{i}_to_{j}' 
                for i in range(5) 
                for j in range(i, 5)]
    args.feature_sets.extend(cci_list)


def main(passed_args=None, dataset=None):
    # Get local rank and world size from environment variables (usually set by the launcher)
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    ## BASIC SETUP ##
    fix_random_seed(args.random_seed)
    
    gpu_setup(args, local_rank, world_size)

    #################
    model = initialize_model(args, local_rank)

    # Define checkpoint path
    model_path = os.path.join(args.model_dir, 'best_checkpoint.pth')

    # Define loss function and optimizer
    loss_fn = get_loss_fn(args.loss_fn_name)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.T_max, eta_min=0)
    model, opt, start_epoch, _ = load_checkpoint(model, opt, model_path, args.device)

    num_list = None if BENCHMARK else [i for i in range(CATALOG_SIZE)]
    train_dataset, val_dataset, test_dataset, common_size = load_and_prepare_data(num_list, args, global_rank, world_size)

    ### NEIGHBORHOOD DROPPING ###
    logging.info(f"Processing Augmentation with Drop Probability {args.drop_prob}")
    for dataset in [test_dataset]:
        if dataset is None:
            continue
        else:
            dataset.augment(args.drop_prob, args.cci_mode)

    # Evaluation - only from rank 0
    if global_rank == 0:
        logging.info("Starting evaluation")
        evaluate(model, test_dataset, args.device, os.path.join(os.path.dirname(args.checkpoint_dir), "pred.txt"), args.target_labels)
    
    ## CLEAN UP ##
    if not args.tuning:
        dist.destroy_process_group()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
