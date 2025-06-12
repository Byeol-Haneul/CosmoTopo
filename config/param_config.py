'''
Author: Jun-Young Lee

This module provides parameter normalization utilities for cosmological simulation datasets. 
This module is relevant for the training/testing phase.

Constants:
- PARAM_STATS: Dict mapping parameter names to min/max ranges.
- PARAM_ORDER: Defines the order in which parameters appear in data arrays.

Functions:
- normalize_params: Normalize physical parameters to [0, 1] based on dataset-specific min/max values.
- denormalize_params: Denormalize predictions from normalized space back to original physical parameter units.
'''

import numpy as np
import torch
from config.machine import BASE_DIR, TYPE

if TYPE == "Quijote":
    PARAM_STATS = {
        "Omega_m": {"min": 0.1, "max": 0.5},
        "Omega_b": {"min": 0.03, "max": 0.07},
        "h": {"min": 0.5, "max": 0.9},
        "n_s": {"min": 0.8, "max": 1.2},
        "sigma_8": {"min": 0.6, "max": 1.0}
    }
    PARAM_ORDER = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
elif TYPE == "CAMELS":
    PARAM_STATS = {
        "Omega0": {"min": 0.1, "max": 0.5},
        "sigma8": {"min": 0.6, "max": 1.0},
        "ASN1": {"min": 0.25, "max": 4.0},
        "AAGN1": {"min": 0.25, "max": 4.0},
        "ASN2": {"min": 0.5, "max": 2.0},
        "AAGN2": {"min": 0.5, "max": 2.0}
    }
    PARAM_ORDER = ["Omega0", "sigma8", "ASN1", "AAGN1", "ASN2", "AAGN2"]
else:
    raise Exception("Invalid Simulation Suite")

def normalize_params(y_list: list[torch.tensor], target_labels: list[str]) -> list[torch.tensor]:
    norm_y_list = []
    for y in y_list: #already sorted by target_labels's order
        norm_y = torch.zeros_like(y)
        for i, param in enumerate(target_labels):
            norm_y[i] = (y[i] - PARAM_STATS[param]["min"]) / (PARAM_STATS[param]["max"] - PARAM_STATS[param]["min"])
        norm_y_list.append(norm_y)
    return norm_y_list


def denormalize_params(norm_y: np.ndarray, target_labels: list[str]) -> np.ndarray:
    num_params = len(target_labels)
    y = np.zeros((norm_y.shape[0], num_params * 2))
    
    for i, param in enumerate(target_labels):
        y[:, i] = norm_y[:, i] * (PARAM_STATS[param]["max"] - PARAM_STATS[param]["min"]) + PARAM_STATS[param]["min"]
    
    if norm_y.shape[1] == num_params * 2: # for inferring first moments (std)
        for i, param in enumerate(target_labels):
            y[:, i + num_params] = norm_y[:, i + num_params] * (PARAM_STATS[param]["max"] - PARAM_STATS[param]["min"])
        return y
    else:
        return y[:,:num_params]
