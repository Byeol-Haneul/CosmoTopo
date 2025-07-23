#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --output=stdout1
#SBATCH --error=stderr1
#SBATCH --ntasks=256
#SBATCH --time=24:00:00
#SBATCH --partition=preempt
#SBATCH --qos=preempt
#SBATCH --constraint=rome
#SBATCH --mem=768G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junyounglee@flatironinstitute.org       

cd $SLURM_SUBMIT_DIR

module load openmpi/4.0.7
source ~/.bashrc
conda activate topo

echo
echo The following nodes will be used to run this program:
echo
echo

# Define arrays
r_links=(0.01 0.015 0.02)
numcuts=(3000 4000 5000)
types=("Quijote")

# Loop over TYPE
for type_name in "${types[@]}"; do
    export TYPE="$type_name"
    echo "========================================================"
    echo "Starting runs for TYPE: $type_name"
    echo "========================================================"

    if [ "$TYPE" == "CAMELS" ]; then
        for r_val in "${r_links[@]}"; do
            export R_LINK="$r_val"
            export NUMCUT=-1
            echo "  --- Running CAMELS with R_LINK=$R_LINK, NUMCUT=$NUMCUT ---"
            mpiexec -n 256 python3 generate_cc.py
        done
    else
        for cut in "${numcuts[@]}"; do
            export R_LINK=0.015
            export NUMCUT="$cut"
            echo "  --- Running Quijote with R_LINK=$R_LINK, NUMCUT=$NUMCUT ---"
            mpiexec -n 256 python3 generate_cc.py
        done
    fi

    echo "========================================================"
    echo "Finished runs for TYPE: $type_name"
    echo "========================================================"
    echo ""
done
