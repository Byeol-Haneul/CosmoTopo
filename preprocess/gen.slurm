#!/bin/bash
#SBATCH --job-name=preprocessing               # Job name
#SBATCH --output=stdout                        # Standard output log
#SBATCH --error=stderr                         # Standard error log
#SBATCH --ntasks=128                           # Number of tasks (Express Serial limit)
#SBATCH --time=6:00:00                         # Time limit (1 day, adjust if needed)
#SBATCH --partition=genx                       # Express Serial partition
#SBATCH --constraint=rome                      # Use 128-core AMD Rome nodes
#SBATCH --mem=128G                             # Request 128GB of memory
#SBATCH --mail-type=ALL                        # Mail notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=toti010@naver.com          # Email address for notifications     

cd $SLURM_SUBMIT_DIR

module load openmpi/4.0.7

source ~/.bashrc
conda activate topo

echo
echo The following nodes will be used to run this program:
echo
srun --ntasks=128 hostname
echo

for r in 0.01 0.02 0.015 ; do
    export R_LINK=$r  # Make R_LINK available to mpiexec
    echo "Running generate_cc.py with R_LINK=$R_LINK"
    mpiexec -n 128 python3 generate_cc.py
done

'''
for nc in 8000 10000; do
    export NUMCUT=$nc  # Make R_LINK available to mpiexec
    echo "Running generate_cc.py with NUMCUT=$NUMCUT"
    mpiexec -n 256 python3 generate_cc.py
done
'''
exit 0
