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
