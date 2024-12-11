'''
Note:
This is a hand-tunable hyperparameter configuration file for testing with a single run, without Optuna. 
Once using Optuna via ~/tune.py, the hyperparameters will be sampled from config/hyperparam.py
'''

import torch
from argparse import Namespace
from config.machine import *

args = Namespace(
    tuning=False,
    only_positions=True,
    
    # Directories
    data_dir=DATA_DIR+"/tensors/",
    checkpoint_dir=RESULT_DIR+"/test/",
    label_filename=LABEL_FILENAME,
    
    # Model Architecture
    in_channels=[1, 3, 5, 7, 3],
    hidden_dim=64,
    num_layers=3,
    layerType="TNN",
    attention_flag=False,
    residual_flag=True,
    
    # Target Labels
    target_labels=["Omega_m", "sigma_8"] if TYPE == "Quijote" else ["Omega0"],
    
    # Training Hyperparameters
    num_epochs=3000,
    test_interval=20,
    T_max=10,
    learning_rate=5e-4,
    weight_decay=1e-5,
    batch_size=64,
    drop_prob=0,
    
    # Device
    device_num="0,1",
    device=None,
    
    # Dataset Split and Random Seed
    val_size=0.1,
    test_size=0.1,
    random_seed=1234,
    
    # Features & Neighborhood Functions
    cci_mode="euclidean",
    
    feature_sets=[
        'x_0', 'x_1', 'x_2', 'x_3', 'x_4',
        f'cci_mode_{cci_mode}_0_to_0', f'cci_mode_{cci_mode}_1_to_1', f'cci_mode_{cci_mode}_2_to_2',
        f'cci_mode_{cci_mode}_3_to_3', f'cci_mode_{cci_mode}_4_to_4',
        f'cci_mode_{cci_mode}_0_to_1', f'cci_mode_{cci_mode}_0_to_2', f'cci_mode_{cci_mode}_0_to_3',
        f'cci_mode_{cci_mode}_0_to_4', f'cci_mode_{cci_mode}_1_to_2',
        f'cci_mode_{cci_mode}_1_to_3', f'cci_mode_{cci_mode}_1_to_4',
        f'cci_mode_{cci_mode}_2_to_3', f'cci_mode_{cci_mode}_2_to_4',
        f'cci_mode_{cci_mode}_3_to_4',
        'global_feature'
    ],
)
