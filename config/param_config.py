import numpy as np
import torch
from config.machine import BASE_DIR, TYPE

if "Quijote" in TYPE:
    PARAM_STATS = {
        "Omega_m": {"min": 0.1, "max": 0.5},
        "Omega_b": {"min": 0.03, "max": 0.07},
        "h": {"min": 0.5, "max": 0.9},
        "n_s": {"min": 0.8, "max": 1.2},
        "sigma_8": {"min": 0.6, "max": 1.0}
    }
    PARAM_ORDER = ["Omega_m", "Omega_b", "h", "n_s", "sigma_8"]
elif "CAMELS-SAM" in TYPE or "CAMELS-TNG" in TYPE:
    PARAM_STATS = {
        "Omega_m": {"min": 0.1, "max": 0.5},
        "sigma_8": {"min": 0.6, "max": 1.0},
        "A_sn1": {"min": 0.25, "max": 4.0},
        "Aagn1": {"min": 0.25, "max": 4.0},
        "A_sn2": {"min": 0.5, "max": 2.0},
    }
    PARAM_ORDER = None
elif TYPE == "fR":
    PARAM_STATS = {
        "Omega_M": {"min": 0.1, "max": 0.5},
        "Omega_b": {"min": 0.03, "max": 0.07},
        "h": {"min": 0.5, "max": 0.9},
        "ns": {"min": 0.8, "max": 1.2},
        "s8(LCDM)": {"min": 0.6, "max": 1.0},
        "m_nu": {"min": 0.01, "max": 1.0},
        "f_R0": {"min": -3e-4, "max": 0},
        "s8(MG)": {"min": 0.6, "max": 1.2}, # Not a Uniform Prior
        "A_s": {"min": -3e-4, "max": 0}     # Not a Uniform Prior
    }
    PARAM_ORDER = ["Omega_M", "Omega_b", "h", "ns", "s8(LCDM)", "m_nu", "f_R0", "s8(MG)", "A_s"]
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
elif TYPE == "CAMELS_50" or TYPE == "CAMELS_SB28":
    PARAM_STATS = {
        "Omega0": {"min": 0.1, "max": 0.5},
        "sigma8": {"min": 0.6, "max": 1.0},
    }
    PARAM_ORDER = ["Omega0", "sigma8"]
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
