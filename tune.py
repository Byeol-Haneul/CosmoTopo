'''
Author: Jun-Young Lee

Summary:
Runs hyperparameter tuning using Optuna for a specified model architecture.
It supports both isolated architecture tuning (GNN, TNNs) and integrated multi-layer tuning.

! Important
- Remember to set config/machine.py to configure simulations and paths.
- Remember to set config/hyperparam.py to configure the hyperparameter space.
- Set only_positions True, unless you want to use other features. 

Notes:
- Use 'All' for integrated runs.
- Use run.slurm to submit tasks for multi-node, multi-gpu inference.
'''

import os
import argparse
from config.hyperparam import HyperparameterTuner, run_optuna_study
from config.machine import *

def tune(layerType):
    data_dir_base = DATA_DIR
    if layerType == "All":
        checkpoint_dir = RESULT_DIR + f"/integrated_{TYPE}/"
        n_trials = 300
    else:
        checkpoint_dir = RESULT_DIR + f"/isolated_{TYPE}_{layerType}/"
        n_trials = 100

    only_positions = True
    device_num = "0,1,2,3"  # Not necessary

    run_optuna_study(data_dir_base, checkpoint_dir, LABEL_FILENAME, device_num, n_trials, only_positions, layerType=layerType)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning script with architecture selection.")
    parser.add_argument(
        "--layerType", 
        type=str, 
        choices=["GNN", "TetraTNN", "ClusterTNN", "TNN", "All"], 
        required=True,
        help="Specify the model architecture for tuning. Choices are 'GNN', 'TetraTNN', 'ClusterTNN', 'TNN', or 'All'."
    )
    args_tune = parser.parse_args()
    tune(args_tune.layerType)
