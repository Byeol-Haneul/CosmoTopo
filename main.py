'''
Author: Jun-Young Lee

Summary:
Main function governing the total train-evaluation phase in multi-node, multi-gpu environment. 

! Important
- When using fixed hyperparameters, run this as main
- When optimizing hyperparameters,  run this via tune.py

Notes:
- Logger configuration
- GPU setup and seed fixing
- Tensor data loading and normalization
- Dataset splitting per rank (train/val/test)
- Model construction and DDP wrapping
- Training, evaluation, and checkpointing
'''

import logging
import os, sys
import random
import pandas as pd
import numpy as np
import datetime

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from data.batch import collate_topological_batch
from data.load_data import load_tensors, split_indices
from data.dataset import CustomDataset

from model.network import Network
from train import train, evaluate
from utils.loss_functions import *

from config.param_config import PARAM_STATS, PARAM_ORDER, normalize_params
from config.machine import *

def setup_logger(log_filename):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

def file_cleanup(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.tuning:
        try:
            os.remove(os.path.join(args.checkpoint_dir, 'model_checkpoint.pth'))
            os.remove(os.path.join(args.checkpoint_dir, 'pred.txt'))
            os.remove(os.path.join(args.checkpoint_dir, 'train_losses.csv'))
            os.remove(os.path.join(args.checkpoint_dir, 'val_losses.csv'))
            os.remove(os.path.join(args.checkpoint_dir, 'training.log'))
        except OSError:
            pass 

    log_filename = os.path.join(args.checkpoint_dir, f'training.log')
    setup_logger(log_filename)

    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    
def gpu_setup(args, local_rank, world_size):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    args.device = torch.device(f"cuda:{local_rank}")   
    torch.cuda.set_device(args.device)
    dist.init_process_group(backend="nccl", init_method='env://', timeout=datetime.timedelta(days=2)) 
    print(f"[GPU SETUP] Process {local_rank} set up on device {args.device}", file = sys.stderr)

def fix_random_seed(seed):
    seed = seed if (seed is not None) else 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seed fixed to {seed}.")

def load_and_prepare_data(args, global_rank, world_size):
    total_samples = CATALOG_SIZE
    num_val_samples = int(args.val_size * total_samples)
    num_test_samples = int(args.test_size * total_samples)
    num_train_samples = total_samples - num_val_samples - num_test_samples
    
    train_indices = list(range(num_train_samples))
    val_indices = list(range(num_train_samples, num_train_samples + num_val_samples))
    test_indices = list(range(num_train_samples + num_val_samples, total_samples))

    if global_rank == 0:
        logging.info(f"Saving data split indices to {args.checkpoint_dir}")
        split_path = os.path.join(args.checkpoint_dir, "split_indices.txt")
        with open(split_path, "w") as f:
            f.write(f"Train Indices ({len(train_indices)}):\n{train_indices}\n\n")
            f.write(f"Validation Indices ({len(val_indices)}):\n{val_indices}\n\n")
            f.write(f"Test Indices ({len(test_indices)}):\n{test_indices}\n\n")
        logging.info(f"Data split indices saved to {split_path}")

    # Split data equally across processes
    train_indices_rank = split_indices(train_indices, global_rank, world_size)
    val_indices_rank = split_indices(val_indices, global_rank, world_size)
    test_indices_rank = test_indices if global_rank == 0 else []

    print(f"[RANK{global_rank}]: has {len(train_indices_rank)} files", file=sys.stdout)
    sys.stdout.flush()

    # Load training tensors
    data_dir, label_filename, target_labels = (
        args.data_dir, args.label_filename, args.target_labels
    )

    logging.info(f"Rank {global_rank}: Loading training tensors for {len(train_indices_rank)} samples.")
    train_tensor_dict = load_tensors(
        train_indices_rank, data_dir, label_filename, args, target_labels
    )

    feature_sets = train_tensor_dict.keys()
    train_tensor_dict['y'] = normalize_params(train_tensor_dict['y'], target_labels)
    train_data = {feature: train_tensor_dict[feature] for feature in feature_sets}
    train_tuples = list(zip(*[train_data[feature] for feature in feature_sets]))
    train_dataset = CustomDataset(train_tuples, feature_sets)

    # Load validation tensors
    logging.info(f"Rank {global_rank}: Loading validation tensors for {len(val_indices_rank)} samples.")
    val_tensor_dict = load_tensors(
        val_indices_rank, data_dir, label_filename, args, target_labels
    )
    val_tensor_dict['y'] = normalize_params(val_tensor_dict['y'], target_labels)
    val_data = {feature: val_tensor_dict[feature] for feature in feature_sets}
    val_tuples = list(zip(*[val_data[feature] for feature in feature_sets]))
    val_dataset = CustomDataset(val_tuples, feature_sets)

    # Load test tensors (only for rank 0)
    if global_rank == 0:
        logging.info(f"Rank {global_rank}: Loading test tensors for {len(test_indices_rank)} samples.")
        test_tensor_dict = load_tensors(test_indices_rank, data_dir, label_filename, args, target_labels)
        test_tensor_dict['y'] = normalize_params(test_tensor_dict['y'], target_labels)
        test_data = {feature: test_tensor_dict[feature] for feature in feature_sets}
        test_tuples = list(zip(*[test_data[feature] for feature in feature_sets]))
        test_dataset = CustomDataset(test_tuples, feature_sets)
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset

def initialize_model(args, local_rank):
    inout_channels = [args.hidden_dim] * len(args.in_channels)
    channels_per_layer = [[args.in_channels, inout_channels]]  # Input layer

    for _ in range(args.num_layers - 1):
        channels_per_layer.append([inout_channels, inout_channels])  # Hidden layers

    logging.info(f"Model architecture: {channels_per_layer}")
    logging.info("Initializing model")
    
    # Define final output layer
    if args.loss_fn_name == "mse":
        final_output_layer = len(args.target_labels)
    else:
        final_output_layer = len(args.target_labels) * 2

    # Initialize the model
    model = Network(args.layerType, channels_per_layer, final_output_layer, args.cci_mode, args.update_func, args.aggr_func, args.residual_flag, args.loss_fn_name)
    model.to(args.device)

    # Only wrap in DDP if there are multiple GPUs
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    return model


def main(passed_args=None, dataset=None):    
    ##########################################################
    ##                      BASIC SETUP                     ##
    ##########################################################

    if passed_args is None:
        from config.config import args  # Import only if not passed
    else:
        args = passed_args

    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if global_rank == 0:
        file_cleanup(args)

    fix_random_seed(args.random_seed)
    
    if not args.tuning:
        gpu_setup(args, local_rank, world_size)


    #############################################################
    ##                DATA LOADING AND AUGMENTATION            ##
    #############################################################
    if dataset is None:
        train_dataset, val_dataset, test_dataset = load_and_prepare_data(args, global_rank, world_size)

    logging.info(f"Processing Augmentation with Drop Probability {args.drop_prob}")
    for dataset in [train_dataset, val_dataset, test_dataset]:
        if dataset is None:
            continue
        else:
            dataset.augment(args.drop_prob, args.cci_mode)

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_topological_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_topological_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_topological_batch) if test_dataset is not None else None

    ##########################################################
    ##                 MODEL TRAINING & EVALUATION          ##
    ##########################################################

    model = initialize_model(args, local_rank)

    loss_fn = get_loss_fn(args.loss_fn_name)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.learning_rate, max_lr=1.e-3, cycle_momentum=False)

    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')

    logging.info("Starting training")
    best_loss = train(model, train_loader, val_loader, test_loader, loss_fn, opt, scheduler, args, checkpoint_path, global_rank)

    if global_rank == 0:
        logging.info("Starting evaluation")
        evaluate(model, test_loader, args.device, os.path.join(os.path.dirname(checkpoint_path), "pred.txt"), args.target_labels)

    ##################################################
    ##                 CLEANUP                      ##
    ##################################################

    if not args.tuning:
        dist.destroy_process_group()

    torch.cuda.empty_cache()

    return best_loss


if __name__ == "__main__":
    main()
