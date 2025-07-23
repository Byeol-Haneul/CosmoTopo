'''
Author: Jun-Young Lee

Summary: Train/Evaluation functions. 

Note:
- Gradient accumulation to bypass memory constraints.
- Save or load training state (model, optimizer, epoch, loss).
- Distributed training enabled.
'''

import torch
import torch.distributed as dist

import numpy as np
import logging
import os, sys
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from contextlib import nullcontext

from config.param_config import PARAM_STATS, PARAM_ORDER, denormalize_params

def save_checkpoint(model, optimizer, epoch, loss, path):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(state, path)
    logging.info(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path, device, eval_mode=False):
    if os.path.isfile(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        epoch = state['epoch']
        loss = state['loss']
        logging.info(f"Checkpoint loaded from {path}")
    else:
        logging.error(f"No checkpoint found at {path}")
        epoch = 0
        loss = float("inf")

    if not eval_mode:
        dist.barrier()  # Ensure all processes reach this point
    
    return model, optimizer, epoch, loss

def data_to_device(data, device):
    return {
        key: tensor.to(device, non_blocking=True) if tensor is not None else None
        for key, tensor in data.items()
    }


def train(model, train_loader, val_loader, test_loader, loss_fn, opt, scheduler, args, checkpoint_path, global_rank):
    num_epochs, test_interval, device = args.num_epochs, args.test_interval, args.device

    start_epoch = 1
    best_validation_loss = float('inf')
    best_checkpoint_path = os.path.join(os.path.dirname(checkpoint_path), "best_checkpoint.pth")

    train_losses = []
    val_losses = []

    if os.path.isfile(checkpoint_path):
        model, opt, start_epoch, _ = load_checkpoint(model, opt, checkpoint_path, device)

    torch.cuda.empty_cache()

    for epoch_i in range(start_epoch, num_epochs + 1):
        epoch_loss = []
        model.train()
        opt.zero_grad()

        '''
        is_profiling_epoch = (epoch_i == 1)

        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=tensorboard_trace_handler('./logdir'),
            record_shapes=True,
            with_stack=True
        ) if is_profiling_epoch else nullcontext()
        '''

        #with profiler_ctx as prof:
        for data_idx, data in enumerate(train_loader):
            
            data = data_to_device(data, device)
            y = data['y']
            
            y_hat = model(data)
            loss = loss_fn(y_hat, y)

            loss.backward()
            opt.step()
            opt.zero_grad()

            epoch_loss.append(loss.item())
            #if is_profiling_epoch:
            #    prof.step()

        local_avg_train_loss = np.mean(epoch_loss)
        train_losses.append(local_avg_train_loss)

        tensor_loss = torch.tensor(local_avg_train_loss, device=device)
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = tensor_loss.item() / dist.get_world_size()            

        val_loss = validate(model, val_loader, loss_fn, device, epoch_i)
        if global_rank == 0:
            logging.info(f"Epoch: {epoch_i}, Train Loss: {avg_train_loss:.6f}")
            logging.info(f"Epoch: {epoch_i}, Validation Loss: {val_loss:.6f}")
            val_losses.append(val_loss)

            save_checkpoint(model, opt, epoch_i, val_loss, checkpoint_path)

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                save_checkpoint(model, opt, epoch_i, best_validation_loss, best_checkpoint_path)

            loss_dir = os.path.dirname(checkpoint_path)
            pd.DataFrame({"train_loss": train_losses}).to_csv(os.path.join(loss_dir, "train_losses.csv"), index=False)
            pd.DataFrame({"val_loss": val_losses}).to_csv(os.path.join(loss_dir, "val_losses.csv"), index=False)

        if epoch_i % test_interval == 0 and global_rank == 0:
            logging.info(f"Starting evaluation for epoch {epoch_i}")
            current_model_state = model.state_dict()
            current_opt_state = opt.state_dict()

            model, opt, _, _ = load_checkpoint(model, opt, best_checkpoint_path, device, eval_mode=True)
            evaluate(model, test_loader, device, os.path.join(os.path.dirname(best_checkpoint_path), "pred.txt"), args.target_labels)

            model.load_state_dict(current_model_state)
            opt.load_state_dict(current_opt_state)

        scheduler.step()
    
    model, opt, _, _ = load_checkpoint(model, opt, best_checkpoint_path, device, eval_mode=True)
    tensor_best_validation_loss = torch.tensor(best_validation_loss, device=device)
    dist.broadcast(tensor_best_validation_loss, src=0)
    best_validation_loss = tensor_best_validation_loss.item()

    return best_validation_loss

def validate(model, val_loader, loss_fn, device, epoch_i):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for data_idx, data in enumerate(val_loader):
            data = data_to_device(data, device)
            y = data['y']
        
            y_hat = model(data)
            loss = loss_fn(y_hat, y)
            val_loss.append(loss.item())
            logging.debug(f"Epoch: {epoch_i}, Validation Iteration: {data_idx + 1}, Loss: {loss.item():.6f}")

    local_avg_val_loss = np.mean(val_loss) if val_loss else 0.0
    tensor_val_loss = torch.tensor(local_avg_val_loss, device=device)

    dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
    avg_val_loss = tensor_val_loss.item() / dist.get_world_size()
    return avg_val_loss

def evaluate(model, test_loader, device, pred_filename, target_labels):
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for data_idx, data in enumerate(test_loader):
            data = data_to_device(data, device)
            y = data['y']
            y_hat = model.module(data) if hasattr(model, 'module') else model(data)
            predictions.extend(y_hat.cpu().numpy())
            real_values.extend(y.cpu().numpy())
            logging.debug(f"Test Iteration: {data_idx + 1}, Real: {y.cpu().numpy()}, Pred: {y_hat.cpu().numpy()}")
    
    denormalized_predictions = denormalize_params(np.array(predictions), target_labels)
    denormalized_real_values = denormalize_params(np.array(real_values), target_labels)

    pred_df = pd.DataFrame({
        "real": list(denormalized_real_values),
        "pred": list(denormalized_predictions)
    })
    pred_df.to_csv(pred_filename, index=False)

    logging.info(f"Predictions saved to {pred_filename}")
