'''
Author: Jun-Young Lee

This file serves as a master configuration for both preprocessing/training/testing phases.

Configuration settings:
    - MACHINE: Specifies the computational environment. Options: "RUSTY", "HAPPINESS".
    - TYPE: Defines the type of simulation dataset.     Options: "Quijote", "CAMELS".
    - SUBGRID: Allows extensibility for different subgrid models in the future. Current option: "IllustrisTNG".
'''

import os 

MACHINE = "RUSTY"
TYPE    = os.getenv("TYPE", "Quijote")
SUBGRID = "IllustrisTNG"

if MACHINE == "HAPPINESS":
    BASE_DIR   = "/data2/jylee/topology/"
    DATA_DIR   = BASE_DIR + "/IllustrisTNG/combinatorial/"
    RESULT_DIR = BASE_DIR + "/IllustrisTNG/combinatorial/results/"
    
elif MACHINE=="RUSTY":
    if TYPE == "Quijote":
        BASE_DIR       = "/mnt/home/jlee2/quijote/"
        DATA_DIR       = "/mnt/home/jlee2/quijote/"
        RESULT_DIR     = "/mnt/home/jlee2/quijote/results/"
        LABEL_FILENAME = BASE_DIR + "sims/latin_hypercube_params.txt"
    elif TYPE == "CAMELS":
        BASE_DIR       = f"/mnt/home/jlee2/camels/{SUBGRID}/"
        DATA_DIR       = f"/mnt/home/jlee2/camels/{SUBGRID}/"
        RESULT_DIR     = f"/mnt/home/jlee2/camels/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L25n256_LH.txt"
    else:
        raise Exception("Invalid Simulation Suite")
else:
    raise Exception("Invalid Machine")

if TYPE == "Quijote":
    CATALOG_SIZE = 2000 
elif TYPE == "CAMELS": 
    CATALOG_SIZE = 1000