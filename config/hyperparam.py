'''
Author: Jun-Young Lee

This script provides an automated, distributed hyperparameter tuning pipeline using 
Optuna and PyTorch Distributed. It is designed for multi-node, multi-GPU training and supports
different simulation datasets and model configurations. 

Main Features:
- Distributed training across GPUs using PyTorch's NCCL backend.
- Hyperparameter optimization with Optuna/Optuna-integration, including custom trial suggestions.
- Result logging, visualization, and best parameter export.
- Heartbeat thread for monitoring long runs.

Class: HyperparameterTuner
Entrypoint: run_optuna_study()
'''

import optuna
import optuna_integration
import torch
import torch.distributed as dist

import time
import os, sys, socket
import threading
import json

from argparse import Namespace
from main import main, load_and_prepare_data
from config.machine import *

class HyperparameterTuner:
    def __init__(self, data_dir_base, checkpoint_dir, label_filename, device_num, only_positions, global_rank, local_rank, world_size, layerType):
        ## Distributed
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.layerType = layerType

        ## Training Config
        self.data_dir_base = data_dir_base
        self.checkpoint_dir = checkpoint_dir
        self.label_filename = label_filename
        self.device_num = device_num
        self.only_positions = only_positions

        # Create a fixed base args for loading data
        self.base_args = self.create_base_args()
        self.dataset   = None 

    def create_base_args(self):
        if TYPE == "Quijote":
            target_labels = ["Omega_m", "sigma_8"]
        elif TYPE == "CAMELS":
            target_labels = ["Omega0"]

        return Namespace(
            # Mode
            tuning=True,
            only_positions=self.only_positions,

            # Model Architecture
            in_channels=[1, 3, 5, 7, 3],
            attention_flag=False,
            residual_flag=True,

            # Target Labels
            target_labels=target_labels,

            # Directories
            checkpoint_dir=self.checkpoint_dir,
            label_filename=self.label_filename,

            # Training Hyperparameters
            num_epochs=300,
            test_interval=100,

            # Device
            device_num=self.device_num,
            device=None,

            # Fixed Values
            val_size=0.1,
            test_size=0.1,
            random_seed=12345,
        )

    def objective(self, single_trial):
        self.gpu_setup()
        trial = optuna_integration.TorchDistributedTrial(single_trial)

        if TYPE == "Quijote":
            data_dir =  self.data_dir_base + trial.suggest_categorical('data_mode', ['tensors_3000', 'tensors_4000', 'tensors_5000'])
        elif TYPE == "CAMELS":
            data_dir = self.data_dir_base + trial.suggest_categorical('data_mode', ['tensors', 'tensors_sparse', 'tensors_dense'])

        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        num_layers = trial.suggest_int('num_layers', 1, 6)
        cci_mode = trial.suggest_categorical('cci_mode', ['euclidean', 'hausdorff'])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        if self.layerType == "All":
            layer_type = trial.suggest_categorical('layerType', ['TetraTNN', 'ClusterTNN', 'TNN', 'GNN'])
        else:
            layer_type = self.layerType

        batch_size = trial.suggest_categorical('batch_size', [1,2,4,8])
        drop_prob = trial.suggest_float('drop_prob', 0, 0.2, log=False)
        T_max = trial.suggest_int('T_max', 10, 100)
        update_func = trial.suggest_categorical('update_func', ['tanh', 'relu'])
        aggr_func = trial.suggest_categorical('aggr_func', ['sum', 'max', 'min', 'all'])

        trial_checkpoint_dir = os.path.join(self.checkpoint_dir, f'trial_{trial.number}')
        os.makedirs(trial_checkpoint_dir, exist_ok=True)

        # Create args with the broadcasted hyperparameters
        self.args = self.create_args(data_dir, trial_checkpoint_dir, 
                                    hidden_dim, num_layers,  cci_mode,
                                    learning_rate, weight_decay, layer_type, batch_size,
                                    drop_prob, update_func, aggr_func, T_max)

        # Train and evaluate
        val_loss = self.train_and_evaluate(self.args)
        return val_loss

    def create_args(self, data_dir, checkpoint_dir, hidden_dim, num_layers, cci_mode, 
                        learning_rate, weight_decay, layer_type, batch_size,
                        drop_prob, update_func, aggr_func, T_max = 30):

        args = Namespace(**self.base_args.__dict__)
        args.data_dir = data_dir
        args.hidden_dim = hidden_dim
        args.num_layers = num_layers
        args.cci_mode = cci_mode
        args.learning_rate = learning_rate
        args.weight_decay = weight_decay
        args.layerType = layer_type
        args.batch_size = batch_size
        args.drop_prob = drop_prob
        args.checkpoint_dir = checkpoint_dir
        args.update_func = update_func
        args.aggr_func = aggr_func
        args.T_max = T_max

        # Features & Neighborhood Functions
        args.feature_sets=[
            'x_0', 'x_1', 'x_2', 'x_3', 'x_4',
            'n0_to_0', 'n1_to_1', 'n2_to_2', 'n3_to_3', 'n4_to_4',
            'n0_to_1', 'n0_to_2', 'n0_to_3', 'n0_to_4',
            'n1_to_2', 'n1_to_3', 'n1_to_4',
            'n2_to_3', 'n2_to_4',
            'n3_to_4',
            'global_feature'
        ]

        if args.cci_mode != 'None':
            cci_list = [f'{args.cci_mode}_{i}_to_{j}' 
                        for i in range(5) 
                        for j in range(i, 5)]
            args.feature_sets.extend(cci_list)

        return args

    def load_data(self):
        num_list = [i for i in range(CATALOG_SIZE)]
        return load_and_prepare_data(num_list, self.base_args, self.global_rank, self.world_size)

    def train_and_evaluate(self, args):
        return main(args, self.dataset)
    
    def gpu_setup(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(i) for i in range(torch.cuda.device_count()))
        print(torch.cuda.device_count(), self.local_rank, self.global_rank, file=sys.stderr)
        self.base_args.device = torch.device(f"cuda:{self.local_rank}")   
        torch.cuda.set_device(self.base_args.device)
        print(f"[GPU SETUP] Process {self.local_rank} set up on device {self.base_args.device}", file = sys.stderr)

def run_heartbeat(interval):
    while True:
        print(f"Heartbeat: System running at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(interval)

def run_optuna_study(data_dir, checkpoint_dir, label_filename, device_num, n_trials=50, only_positions=True, heartbeat_interval=1200, study_name="my_study", layerType="All"):
    global_rank = int(os.environ['RANK'])    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend="nccl", init_method='env://') 
    tuner = HyperparameterTuner(data_dir, checkpoint_dir, label_filename, device_num, only_positions, global_rank, local_rank, world_size, layerType)
    study = None
    
    if global_rank == 0:
        heartbeat_thread = threading.Thread(target=run_heartbeat, args=(heartbeat_interval,))
        heartbeat_thread.daemon = True
        heartbeat_thread.start()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(checkpoint_dir, f"optuna_results_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        db_url = f'sqlite:///{os.path.join(checkpoint_dir, "optuna_study.db")}'

        sampler = optuna.samplers.TPESampler(seed=tuner.base_args.random_seed)
        study = optuna.create_study(direction='minimize', sampler=sampler, storage=db_url, study_name=study_name, load_if_exists=True)
        study.optimize(tuner.objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            tuner.objective(None)

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
