import os
import sys
import h5py 
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.machine import *

'''
Note:
Before running preprocessing scripts, please be aware that the master settings (MACHINE, TYPE, SUBGRID) are appropriately set.
'''

# ---- CONSTANTS ---- #
DIM = 3

if "Quijote" in TYPE or TYPE == "fR":
    BOXSIZE = 1e3 
elif "CAMELS-SAM" in TYPE:
    BOXSIZE = 1e2
elif "CAMELS-TNG" in TYPE:
    BOXSIZE = 25
elif TYPE == "CAMELS" or TYPE == "CAMELS_SB28":
    BOXSIZE = 25e3
elif TYPE == "CAMELS_50":
    BOXSIZE = 50e3
else:
    raise Exception("Invalid Simulation Suite: ", TYPE)

if TYPE == "CAMELS" or TYPE == "CAMELS_SB28":
    MASS_UNIT = 1e10
    Nstar_th = 20 
    MASS_CUT = 2e8
elif TYPE == "CAMELS_50":
    MASS_UNIT = 1e10
    Nstar_th = 40
    MASS_CUT = 4e8

modes = {"ISDISTANCE": 1, "ISAREA": 2, "ISVOLUME": 3}
global_centroid = None # to be updated.

# --- HYPERPARAMS --- #
#dense: 0.02, sparse: 0.01, fiducial: 0.015
#r_link = 0.015 
r_link = float(os.getenv("R_LINK", 0.015))
NUMCUT = int(os.getenv("NUMCUT", 5000))
print(f"THE R_LINK IS SET AS {r_link}", file=sys.stderr)
print(f"THE NUMCUT IS SET AS {NUMCUT}", file=sys.stderr)

MINCLUSTER = 5 
NUMPOINTS  = -1
NUMEDGES   = NUMCUT if (TYPE == "CAMELS_50" or TYPE == "CAMELS_SB28") else -1
NUMTETRA   = NUMCUT if ("Quijote" in TYPE or TYPE == "fR" or TYPE == "CAMELS_50" or TYPE == "CAMELS_SB28") else -1

## OPTIONS ##
FLAG_HIGHER_ORDER = False
ENABLE_PROFILING = False
#############

if  TYPE == "fR":# or TYPE == "CAMELS_50" or TYPE == "Quijote" or TYPE == "Bench_Quijote_Coarse_Small":
    in_dir = BASE_DIR + "sims/"
    cc_dir = DATA_DIR + f"cc_{NUMTETRA}/"
    tensor_dir = DATA_DIR + f"tensors_{NUMTETRA}/"
elif TYPE == "CAMELS_SB28":
    in_dir = BASE_DIR + f"sims/"
    cc_dirs_option = {0.03: "cc/", 0.05: "cc_dense/", 0.04: "cc_sparse/"}
    tensor_dirs_option = {0.03: "tensors/", 0.05: "tensors_dense/", 0.04: "tensors_sparse/"}
    cc_dir = DATA_DIR + cc_dirs_option.get(r_link, "")
    tensor_dir = DATA_DIR + tensor_dirs_option.get(r_link, "")
else:
    in_dir = BASE_DIR + f"sims/"
    cc_dirs_option = {0.015: "cc/", 0.02: "cc_dense/", 0.01: "cc_sparse/"}
    tensor_dirs_option = {0.015: "tensors/", 0.02: "tensors_dense/", 0.01: "tensors_sparse/"}
    cc_dir = DATA_DIR + cc_dirs_option.get(r_link, "")
    tensor_dir = DATA_DIR + tensor_dirs_option.get(r_link, "")

os.makedirs(cc_dir, exist_ok=True)
os.makedirs(tensor_dir, exist_ok=True)

def load_catalog(num):
    '''
    Modified from CosmoGraphNet
    arXiv:2204.13713
    https://github.com/PabloVD/CosmoGraphNet/
    '''
    if BENCHMARK:
        f = h5py.File(HDF5_DATA_FILE, 'r')
        if "Quijote" in TYPE:
            data = f['BSQ'][f'BSQ_{num}']
        elif "CAMELS" in TYPE:
            data = f['LH'][f'LH_{num}']

        X = data['X'][:]
        Y = data['Y'][:]
        Z = data['Z'][:]

        VX = data['VX'][:]
        VY = data['VY'][:]
        VZ = data['VZ'][:]
        pos = np.vstack((X,Y,Z)).T / BOXSIZE
        vel = np.stack((VX, VY, VZ)).T
        f.close()
    elif TYPE == "Quijote" or TYPE == "fR":
        in_filename = f"catalog_{num}.txt"
        pos = np.loadtxt(in_dir + in_filename)/BOXSIZE
    else:
        in_filename = f"data_{num}.hdf5"
        f = h5py.File(in_dir + in_filename, 'r')
        pos   = f['/Subhalo/SubhaloPos'][:]/BOXSIZE
        Mstar = f['/Subhalo/SubhaloMassType'][:,4] * MASS_UNIT
        Rstar = f["Subhalo/SubhaloHalfmassRadType"][:,4]
        Nstar = f['/Subhalo/SubhaloLenType'][:,4] 
        Metal = f["Subhalo/SubhaloStarMetallicity"][:]
        Vmax = f["Subhalo/SubhaloVmax"][:]
        f.close()
    
    # Some simulations are slightly outside the box, correct it
    pos[np.where(pos<0.0)]+=1.0
    pos[np.where(pos>1.0)]-=1.0

    if BENCHMARK or "Quijote" in TYPE or TYPE == "fR":
        feat = np.concatenate([pos, vel], axis=1)
    elif "Quijote" in TYPE or TYPE == "fR":
        feat = pos #np.zeros(pos.shape)
    else:
        indexes = np.where(Nstar>Nstar_th)[0]
        #indexes = np.where(Mstar>MASS_CUT)[0]
        pos     = pos[indexes]
        Mstar   = Mstar[indexes]
        Rstar   = Rstar[indexes]
        Metal   = Metal[indexes]
        Vmax    = Vmax[indexes]

        #Normalization
        Mstar = np.log10(1.+ Mstar)
        Rstar = np.log10(1.+ Rstar)
        Metal = np.log10(1.+ Metal)
        Vmax  = np.log10(1.+ Vmax)

        feat = np.vstack((Mstar, Rstar, Metal, Vmax)).T

    return pos, feat

def normalize(value, option):
    power = modes[option]
    return value / (r_link ** power)

def get_splits():
    modes = ["train", "val", "test"]
    split_dict = {}

    for mode in modes:
        suffix = f"_{mode}_lhs.hdf5" if "CAMELS-SAM" in TYPE else f"_{mode}.hdf5"
        filename = os.path.join(BENCH_PATH, TYPE.split('Subset_')[-1] + suffix)

        with h5py.File(filename, "r") as f:
            key_group = "LH" if "CAMELS" in TYPE else "BSQ"
            #num_list = list(f[key_group].keys())
            #nums = [int(k.split("_")[-1]) for k in num_list]
            #split_dict[mode] = np.array(nums)
            numList = f['original_ids'][:]
            split_dict[mode] = np.array(numList)

    os.makedirs(BASE_DIR, exist_ok=True)
    np.savez(os.path.join(BASE_DIR, "splits.npz"), **split_dict)