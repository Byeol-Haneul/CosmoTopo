import logging
import torch
import os, sys
import random
import pandas as pd
import numpy as np
import socket 

from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from data.load_data import load_tensors, split_data
from model.network import Network
from model.train import train, evaluate, save_checkpoint, load_checkpoint
from data.dataset import CustomDataset

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
    dist.init_process_group(backend="nccl", init_method='env://') 
    print(f"[GPU SETUP] Process {local_rank} set up on device {args.device}", file = sys.stderr)

def fix_random_seed(seed):
    seed = seed if (seed is not None) else 12345
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seed fixed to {seed}.")

def implicit_likelihood_loss(output, target):
    num_params = len(target)
    y_out, err_out = output[:,:num_params], output[:,num_params:]
    loss_mse = torch.mean(torch.sum((y_out - target)**2., axis=1), axis=0)
    loss_ili = torch.mean(torch.sum(((y_out - target)**2. - err_out**2.)**2., axis=1), axis=0)
    loss = loss_mse + loss_ili
    return loss

def load_and_prepare_data(num_list, args, global_rank, world_size):
    # Determine the total number of samples
    total_samples = len(num_list)

    # Calculate the number of samples for validation and test
    num_val_samples = int(args.val_size * total_samples)
    num_test_samples = int(args.test_size * total_samples)

    # Calculate the number of training samples
    num_train_samples = total_samples - num_val_samples - num_test_samples

    # Split indices for validation and test
    val_indices = list(range(num_train_samples, num_train_samples + num_val_samples))
    test_indices = list(range(num_train_samples + num_val_samples, total_samples))

    # Prepare data for each rank
    per_process_train_size = num_train_samples // world_size
    start_train_idx = global_rank * per_process_train_size
    end_train_idx = (global_rank + 1) * per_process_train_size

    # Prepare training indices
    train_indices = list(range(start_train_idx, end_train_idx))

    data_dir, label_filename, target_labels, feature_sets = (
        args.data_dir, args.label_filename, args.target_labels, args.feature_sets
    )

    # Load tensors only for the designated train indices
    logging.info(f"Loading training tensors for {len(train_indices)} samples from {data_dir}")
    train_tensor_dict = load_tensors(
        [num_list[i] for i in train_indices], data_dir, label_filename, args, target_labels, feature_sets
    )

    logging.info("Normalizing target parameters for training")
    train_tensor_dict['y'] = normalize_params(train_tensor_dict['y'], target_labels)

    # Prepare training data for all ranks
    train_data = {feature: train_tensor_dict[feature] for feature in feature_sets + ['y']}
    train_tuples = list(zip(*[train_data[feature] for feature in feature_sets + ['y']]))

    logging.info(f"Created train dataset with {len(train_tuples)} samples")
    train_dataset = CustomDataset(train_tuples, feature_sets + ['y'])

    # Divide validation set across ranks
    per_process_val_size = num_val_samples // world_size
    start_val_idx = global_rank * per_process_val_size
    end_val_idx = (global_rank + 1) * per_process_val_size
    val_indices_rank = val_indices[start_val_idx:end_val_idx]

    # Load tensors for the validation subset of each rank
    logging.info(f"Loading validation tensors for {len(val_indices_rank)} samples from {data_dir}")
    val_tensor_dict = load_tensors(
        [num_list[i] for i in val_indices_rank], data_dir, label_filename, args, target_labels, feature_sets
    )
    val_tensor_dict['y'] = normalize_params(val_tensor_dict['y'], target_labels)

    val_data = {feature: val_tensor_dict[feature] for feature in feature_sets + ['y']}
    val_tuples = list(zip(*[val_data[feature] for feature in feature_sets + ['y']]))

    logging.info(f"Created validation dataset with {len(val_tuples)} samples")
    val_dataset = CustomDataset(val_tuples, feature_sets + ['y'])

    # Only rank 0 loads and handles the test set
    if global_rank == 0:
        logging.info(f"Loading test tensors for {len(test_indices)} samples from {data_dir}")
        test_tensor_dict = load_tensors(
            [num_list[i] for i in test_indices], data_dir, label_filename, args, target_labels, feature_sets
        )
        test_tensor_dict['y'] = normalize_params(test_tensor_dict['y'], target_labels)

        test_data = {feature: test_tensor_dict[feature] for feature in feature_sets + ['y']}
        test_tuples = list(zip(*[test_data[feature] for feature in feature_sets + ['y']]))

        logging.info(f"Created test dataset with {len(test_tuples)} samples")
        test_dataset = CustomDataset(test_tuples, feature_sets + ['y'])
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
    final_output_layer = len(args.target_labels) * 2

    # Initialize the model
    model = Network(args.layerType, channels_per_layer, final_output_layer, args.cci_mode, args.update_func, args.aggr_func, args.residual_flag)

    # Move the model to the correct device before wrapping in DDP
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

    num_list = [i for i in range(CATALOG_SIZE)]

    if dataset is None:
        train_dataset, val_dataset, test_dataset = load_and_prepare_data(num_list, args, global_rank, world_size)

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
    loss_fn = implicit_likelihood_loss
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.T_max, eta_min=0)

    # Define checkpoint path
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_checkpoint.pth')

    # Training
    logging.info("Starting training")
    best_loss = train(model, train_dataset, val_dataset, test_dataset, loss_fn, opt, scheduler, args, checkpoint_path, global_rank)
    
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
