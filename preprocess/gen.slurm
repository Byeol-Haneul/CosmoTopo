#!/bin/bash
#SBATCH --job-name=preprocessing               
#SBATCH --output=stdout1                        
#SBATCH --error=stderr1                        
#SBATCH --ntasks=128                           
#SBATCH --time=24:00:00                         
#SBATCH --partition=genx                       
#SBATCH --constraint=rome                      
#SBATCH --mem=128G                             
#SBATCH --mail-type=ALL                       
#SBATCH --mail-user=toti010@naver.com          

cd $SLURM_SUBMIT_DIR

module load openmpi/4.0.7

source ~/.bashrc
conda activate topo

echo
echo The following nodes will be used to run this program:
echo
srun --ntasks=128 hostname
echo

mpiexec -n 128 python3 generate_cc.py

exit 0
