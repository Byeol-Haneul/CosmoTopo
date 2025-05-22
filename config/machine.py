# Configuration settings:
# - MACHINE: Specifies the computational environment. Options: "RUSTY", "HAPPINESS".
# - TYPE: Defines the type of simulation dataset.     Options: "Quijote", "CAMELS".
# - SUBGRID: Allows extensibility for different subgrid models in the future. Current option: "IllustrisTNG".

MACHINE = "RUSTY"
TYPE    = "Quijote_BSQ_rockstar_10_top5000"
SUBGRID = "IllustrisTNG"

BENCHMARK = TYPE in [
    "Subset_Quijote_BSQ_rockstar_10_top5000",
    "Quijote_BSQ_rockstar_10_top5000",
    "CAMELS-SAM_LH_rockstar_99_top5000",
    "CAMELS-SAM_LH_gal_99_top5000",
    "CAMELS-TNG_galaxy_90_ALL",
    "tpcf_Quijote_BSQ_top5000",
    "tpcf_CAMELS-SAM_LH_gal_top5000_mstar_tpcf",
    "tpcf_CAMELS-TNG_galaxy_90_ALL_mstar_tpcf",
]

BENCH_PATH = "/mnt/home/rstiskalek/ceph/graps4science/"

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
    elif TYPE == "Bench_Quijote_Coarse_Small":
        BASE_DIR       = "/mnt/home/jlee2/ceph/benchmark/coarse_small/"
        DATA_DIR       = "/mnt/home/jlee2/ceph/benchmark/coarse_large/"
        RESULT_DIR     = "/mnt/home/jlee2/ceph/benchmark/coarse_small/results/"
        LABEL_FILENAME = BASE_DIR + "BSQ_params.txt"
    elif TYPE == "fR":
        BASE_DIR       = "/mnt/home/jlee2/ceph/fR/"
        DATA_DIR       = "/mnt/home/jlee2/ceph/fR/"
        RESULT_DIR     = "/mnt/home/jlee2/ceph/fR/results/"
        LABEL_FILENAME = BASE_DIR + "full_s8_table.dat"
    elif TYPE == "CAMELS":
        BASE_DIR       = f"/mnt/home/jlee2/ceph/camels/LH/{SUBGRID}/"
        DATA_DIR       = f"/mnt/home/jlee2/ceph/camels/LH/{SUBGRID}/"
        RESULT_DIR     = f"/mnt/home/jlee2/ceph/camels/LH/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L25n256_LH.txt"
    elif TYPE == "CAMELS_50":
        BASE_DIR       = f"/mnt/home/jlee2/ceph/camels/SB35/{SUBGRID}/"
        DATA_DIR       = f"/mnt/home/jlee2/ceph/camels/SB35/{SUBGRID}/"
        RESULT_DIR     = f"/mnt/home/jlee2/ceph/camels/SB35/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L50n512_SB35.txt"
    elif TYPE == "CAMELS_SB28":
        BASE_DIR       = f"/mnt/home/jlee2/ceph/camels/SB28/{SUBGRID}/"
        DATA_DIR       = f"/mnt/home/jlee2/ceph/camels/SB28/{SUBGRID}/"
        RESULT_DIR     = f"/mnt/home/jlee2/ceph/camels/SB28/{SUBGRID}/results/{SUBGRID}/"
        LABEL_FILENAME = BASE_DIR + f"CosmoAstroSeed_{SUBGRID}_L25n256_SB28.txt"
    elif TYPE == "Bench_Quijote_Coarse_Large":
        BASE_DIR       = "/mnt/home/jlee2/ceph/benchmark/coarse_large/"
        DATA_DIR       = "/mnt/home/jlee2/ceph/benchmark/coarse_large/"
        HDF5_DATA_FILE = "/mnt/home/jlee2/ceph/benchmark/coarse_large/Quijote_BSQ_rockstar_10_top5000.hdf5"
        RESULT_DIR     = "/mnt/home/jlee2/ceph/benchmark/coarse_large/results/"
        LABEL_FILENAME = BASE_DIR + "BSQ_params.txt"

        ############################# BENCHMARK ###############################
    elif BENCHMARK:  
        BASE_DIR       = f"/mnt/home/jlee2/ceph/benchmark/{TYPE}/"
        DATA_DIR       = f"/mnt/home/jlee2/ceph/benchmark/{TYPE.split('Subset_')[-1]}/"
        HDF5_DATA_FILE = f"/mnt/home/rstiskalek/ceph/graps4science/{TYPE.split('Subset_')[-1]}.hdf5"
        RESULT_DIR     = BASE_DIR + "results/"
    else:
        raise Exception("Invalid Simulation Suite")
else:
    raise Exception("Invalid Machine")

catalog_sizes = {
    "Quijote": 2000,
    "Bench_Quijote_Coarse_Small": 3072,
    "Bench_Quijote_Coarse_Large": 32768,
    "fR": 2048,
    "CAMELS_SB28": 2048,
    "CAMELS": 1000,
    "CAMELS_50": 1024,


    ## BENCHMARKS ##
    "Subset_Quijote_BSQ_rockstar_10_top5000": 3072,
    "Quijote_BSQ_rockstar_10_top5000": 32752,
    "CAMELS-SAM_LH_rockstar_99_top5000": 1000,
    "CAMELS-SAM_LH_gal_99_top5000": 1000,
    "CAMELS-TNG_galaxy_90_ALL": 1000,

    "tpcf_Quijote_BSQ_top5000": 32752,
    "tpcf_CAMELS-SAM_LH_gal_top5000_mstar_tpcf": 1000,
    "tpcf_CAMELS-TNG_galaxy_90_ALL_mstar_tpcf": 1000,
}

CATALOG_SIZE = catalog_sizes.get(TYPE)

if CATALOG_SIZE is None:
    raise ValueError(f"Unknown TYPE: {TYPE}")

try:
    LABEL_FILENAME
except NameError:
    LABEL_FILENAME = None

if "tpcf" in TYPE:
    TPCF = True
else:
    TPCF = False