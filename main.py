import logging
import os, sys
import random
import pandas as pd
import numpy as np
import socket 
import datetime

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
    '''
    if socket.gethostname() == "node14":
        os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    '''
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

def load_and_prepare_data(num_list, args, global_rank, world_size):
    total_samples = CATALOG_SIZE

    # Handle Bench_Quijote_Coarse_Small case with fixed indices
    if BENCHMARK:
        splits = np.load(BASE_DIR + "splits.npz")
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]
        assert len(train_indices) + len(val_indices) + len(test_indices) == total_samples

    elif TYPE == "Bench_Quijote_Coarse_Small":
        if total_samples != 3072:
            raise ValueError(f"Expected 3072 samples, but got {total_samples}")
        train_indices = list(range(2048))  # First 2048 for training
        val_indices = list(range(2048, 2560))  # Next 512 for validation
        test_indices = list(range(2560, 3072))  # Last 512 for test
    elif TYPE == "fR":
        if total_samples != 2048:
            raise ValueError(f"Expected 2048 samples, but got {total_samples}")
        train_indices = list(range(1536))  # First 2048 for training
        val_indices = list(range(1536, 1792))  # Next 512 for validation
        test_indices = list(range(1792, 2048))  # Last 512 for test
    elif TYPE == "Bench_Quijote_Coarse_Large":
        if total_samples != 32768:
            raise ValueError(f"Expected 32768 samples, but got {total_samples}")
    
        train_indices = list(range(24576))  
        val_indices = list(range(24576, 28672)) 
        test_indices = list(range(28672, 32768)) 
    else:
        # Default dynamic case
        num_val_samples = int(args.val_size * total_samples)
        num_test_samples = int(args.test_size * total_samples)
        num_train_samples = total_samples - num_val_samples - num_test_samples
        
        train_indices = list(range(num_train_samples))
        val_indices = list(range(num_train_samples, num_train_samples + num_val_samples))
        test_indices = list(range(num_train_samples + num_val_samples, total_samples))

    if TYPE == "CAMELS_50":
        # Skip simulation 425, which is not finished as of now. 
        try:
            train_indices.remove(425)
            test_indices.remove(425)
            val_indices.remove(425)
        except:
            pass

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


    common_size = len(train_indices) // world_size

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

    # Load training tensors
    data_dir, label_filename, target_labels, feature_sets = (
        args.data_dir, args.label_filename, args.target_labels, args.feature_sets
    )

    logging.info(f"Rank {global_rank}: Loading training tensors for {len(train_indices_rank)} samples.")
    train_tensor_dict = load_tensors(
        train_indices_rank, data_dir, label_filename, args, target_labels, feature_sets
    )
    train_tensor_dict['y'] = normalize_params(train_tensor_dict['y'], target_labels)
    train_data = {feature: train_tensor_dict[feature] for feature in feature_sets + ['y']}
    train_tuples = list(zip(*[train_data[feature] for feature in feature_sets + ['y']]))
    train_dataset = CustomDataset(train_tuples, feature_sets + ['y'])

    # Load validation tensors
    logging.info(f"Rank {global_rank}: Loading validation tensors for {len(val_indices_rank)} samples.")
    val_tensor_dict = load_tensors(
        val_indices_rank, data_dir, label_filename, args, target_labels, feature_sets
    )
    val_tensor_dict['y'] = normalize_params(val_tensor_dict['y'], target_labels)
    val_data = {feature: val_tensor_dict[feature] for feature in feature_sets + ['y']}
    val_tuples = list(zip(*[val_data[feature] for feature in feature_sets + ['y']]))
    val_dataset = CustomDataset(val_tuples, feature_sets + ['y'])

    # Load test tensors (only for rank 0)
    if global_rank == 0:
        logging.info(f"Rank {global_rank}: Loading test tensors for {len(test_indices_rank)} samples.")
        test_tensor_dict = load_tensors(
            test_indices_rank, data_dir, label_filename, args, target_labels, feature_sets
        )
        test_tensor_dict['y'] = normalize_params(test_tensor_dict['y'], target_labels)
        test_data = {feature: test_tensor_dict[feature] for feature in feature_sets + ['y']}
        test_tuples = list(zip(*[test_data[feature] for feature in feature_sets + ['y']]))
        test_dataset = CustomDataset(test_tuples, feature_sets + ['y'])
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset, common_size

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
    if passed_args is None:
        from config.config import args  # Import only if not passed
    else:
        args = passed_args
    
    # Get local rank and world size from environment variables (usually set by the launcher)
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    ## BASIC SETUP ##
    if global_rank == 0:
        file_cleanup(args)

    fix_random_seed(args.random_seed)
    
    if not args.tuning:
        gpu_setup(args, local_rank, world_size)
        
    num_list = None if BENCHMARK else [i for i in range(CATALOG_SIZE)]

    if dataset is None:
        train_dataset, val_dataset, test_dataset, common_size = load_and_prepare_data(num_list, args, global_rank, world_size)

    ### NEIGHBORHOOD DROPPING ###
    logging.info(f"Processing Augmentation with Drop Probability {args.drop_prob}")
    for dataset in [train_dataset, val_dataset, test_dataset]:
        if dataset is None:
            continue
        else:
            dataset.augment(args.drop_prob, args.cci_mode)

    #################
    model = initialize_model(args, local_rank)

    # Define loss function and optimizer
    loss_fn = get_loss_fn(args.loss_fn_name)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.T_max, eta_min=0)

    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')

    # Training
    logging.info("Starting training")
    best_loss = train(model, train_dataset, val_dataset, test_dataset, loss_fn, opt, scheduler, args, checkpoint_path, global_rank, common_size)
    
    # Evaluation - only from rank 0
    if global_rank == 0:
        logging.info("Starting evaluation")
        evaluate(model, test_dataset, args.device, os.path.join(os.path.dirname(checkpoint_path), "pred.txt"), args.target_labels)
    
    ## CLEAN UP ##
    if not args.tuning:
        dist.destroy_process_group()

    torch.cuda.empty_cache()
    return best_loss

if __name__ == "__main__":
    main()
