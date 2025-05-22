import os
import shutil
import h5py
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import logging
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import optuna, optuna_integration

from config.machine import *  # adjust as needed
from config.param_config import normalize_params, denormalize_params
from main import fix_random_seed
from train import save_checkpoint, load_checkpoint

# Setup global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ---------- Dataset ----------
class TPCFDataset(Dataset):
    def __init__(self, input_file, target_labels):
        super().__init__()
        with h5py.File(input_file, 'r') as f_in:
            self.tpcf = torch.abs(torch.tensor(f_in['tpcf'][:], dtype=torch.float32))
            self.inputs = self.tpcf
            labels = [f_in['params'][param][:] for param in target_labels]
            labels = np.stack(labels, axis=-1) 
            labels = torch.tensor(labels, dtype=torch.float32)
            self.labels = normalize_params(labels, target_labels)

    def __len__(self):
        return self.tpcf.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# ---------- Model ----------
class TPCF_NN(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        layers = []
        for h in hidden_dims:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            input_dim = h
        layers.append(nn.Linear(input_dim, 2))  # Omega_m, sigma_8
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------- Training Function ----------
def train(global_rank, local_rank, world_size, args):
    # load train and val
    train_ds = TPCFDataset(args['train_input'], args['target_labels'])
    val_ds   = TPCFDataset(args['val_input'],   args['target_labels'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, world_size, global_rank)
    val_sampler   = torch.utils.data.distributed.DistributedSampler(val_ds,   world_size, global_rank)

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], sampler=train_sampler)
    val_loader   = DataLoader(val_ds,   batch_size=args['batch_size'], sampler=val_sampler)

    model = TPCF_NN(train_ds.inputs.shape[1], args['hidden_dims'], args['dropout']).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()

    best_val = float('inf')
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(args['epochs']):
        model.train()
        train_sampler.set_epoch(epoch)
        running_train = 0.0
        for x, y in train_loader:
            x, y = x.to(local_rank), y.to(local_rank)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * x.size(0)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(local_rank), y.to(local_rank)
                running_val += criterion(model(x), y).item() * x.size(0)

        if global_rank == 0:
            avg_train = running_train / len(train_loader.dataset)
            avg_val   = running_val   / len(val_loader.dataset)

            train_losses.append(avg_train)
            val_losses.append(avg_val)

            logging.info(f"Epoch {epoch}: Train {avg_train:.4f}, Val {avg_val:.4f}")

            curr_path = os.path.join(args['trial_dir'], 'current_model.pth')
            save_checkpoint(model, optimizer, epoch, avg_val, curr_path)

            # Save best model so far
            if avg_val < best_val:
                best_val = avg_val
                best_model_path = os.path.join(args['trial_dir'], 'best_checkpoint.pth')
                save_checkpoint(model, optimizer, epoch, best_val, best_model_path)

    # After training, restore the best model
    if global_rank == 0:
        best_model_path = os.path.join(args['trial_dir'], 'best_checkpoint.pth')
        model, _, _, _ = load_checkpoint(model, optimizer, best_model_path, device="cuda:0", eval_mode = True)

    return model, train_losses, val_losses, best_val

# ---------- Evaluation Function ----------
def evaluate(model, pred_filename, args):
    ds = TPCFDataset(args['test_input'], args['target_labels'])

    target_labels = args['target_labels']
    loader = DataLoader(ds, batch_size=args['batch_size'], shuffle=False)    
    model.eval()

    preds, reals = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(0)
            y_hat = model(x)
            preds.extend(y_hat.cpu().numpy())
            reals.extend(y.cpu().numpy())

    denormalized_real_values = denormalize_params(np.array(reals), target_labels)
    denormalized_predictions = denormalize_params(np.array(preds), target_labels)
           
    pred_df = pd.DataFrame({
        "real": list(denormalized_real_values),
        "pred": list(denormalized_predictions)
    })
    pred_df.to_csv(pred_filename, index=False)
    logging.info(f"Predictions saved to {pred_filename}")

# ---------- Wrapper for DDP spawn ----------
def train_wrapper(args, trial_num):
    global_rank = int(os.environ['RANK'])    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Setup trial directory
    trial_dir = os.path.join(RESULT_DIR, f'trial_{trial_num}')
    args['trial_dir'] = trial_dir
    
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, 'training.log')
    fh = logging.FileHandler(log_path)
    logging.getLogger().addHandler(fh)

    logging.info(f"Starting trial {trial_num} on rank {local_rank}")

    if global_rank == 0:
        logging.info(f"Trial parameters: {args}")

    # Copy the script to the trial directory
    model, train_loss, val_loss, best_val = train(global_rank, local_rank, world_size, args)
    
    if global_rank == 0:
        # save current and best models
        pd.DataFrame({"train_loss": train_loss}).to_csv(os.path.join(trial_dir, "train_losses.csv"), index=False)
        pd.DataFrame({"val_loss": val_loss}).to_csv(os.path.join(trial_dir, "val_losses.csv"), index=False)
        # evaluate and save predictions
        pred_filename = os.path.join(trial_dir, 'pred.txt')
        evaluate(model, pred_filename, args)
    
    best_val = torch.tensor(best_val, device=local_rank)
    dist.broadcast(best_val, src=0)
    return best_val.item()

# ---------- Optuna Objective ----------
def objective(trial):
    trial = optuna_integration.TorchDistributedTrial(trial)
    args = {
        'train_input': HDF5_DATA_FILE.removesuffix(".hdf5")+"_train.hdf5",
        'val_input':   HDF5_DATA_FILE.removesuffix(".hdf5")+"_val.hdf5",
        'test_input':  HDF5_DATA_FILE.removesuffix(".hdf5")+"_test.hdf5",
        'target_labels': ['Omega_m', 'sigma_8'],
        'batch_size': trial.suggest_categorical('batch_size', [4, 16, 64]),
        'epochs': 300,
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'hidden_dims': [
            trial.suggest_int('h1', 64, 128),
            trial.suggest_int('h2', 64, 128),
            trial.suggest_int('h3', 16, 64),
        ],
        'random_seed': 12345,
    }

    fix_random_seed(args["random_seed"])
    val_loss = train_wrapper(args, trial.number)
    return val_loss

# ---------- Main ----------
if __name__ == '__main__':
    global_rank = int(os.environ['RANK'])    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend="nccl", init_method='env://') 

    n_trials = 100

    if global_rank == 0:
        os.makedirs(RESULT_DIR, exist_ok=True)
        db_url = f'sqlite:///{os.path.join(RESULT_DIR, "optuna_study.db")}'
        study_name = "no-name-f9084c41-85ba-404f-98d4-af0743aaad48"
        sampler = optuna.samplers.TPESampler(seed=12345)
        study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name, storage=db_url, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            objective(None)

    print('Best Trial Params:', study.best_trial.params)

    if global_rank == 0:
        assert study is not None

        best_params_path = os.path.join(result_dir, 'best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)

        trials_df = study.trials_dataframe()
        trials_csv_path = os.path.join(result_dir, 'trials.csv')
        trials_df.to_csv(trials_csv_path, index=False)

        optuna.visualization.matplotlib.plot_optimization_history(study).figure.savefig(os.path.join(result_dir, 'optimization_history.png'))
        optuna.visualization.matplotlib.plot_param_importances(study).figure.savefig(os.path.join(result_dir, 'param_importances.png'))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study).figure.savefig(os.path.join(result_dir, 'parallel_coordinate.png'))

        print(f"Results saved to: {result_dir}")
        print("Best Hyperparameters:", study.best_params)

    dist.destroy_process_group()
