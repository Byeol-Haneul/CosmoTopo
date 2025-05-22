#!/bin/bash
#SBATCH --job-name=run                         
#SBATCH --output=stdout                    
#SBATCH --error=stderr                      
#SBATCH --nodes=8                           
#SBATCH --ntasks=8                           
#SBATCH --cpus-per-task=48                     
#SBATCH --time=120:00:00                       
#SBATCH --partition=gpupreempt                 
#SBATCH --qos=gpupreempt                       
#SBATCH --gpus=a100-sxm4-40gb:32               
#SBATCH --mem=128G                             
#SBATCH --mail-type=ALL                        
#SBATCH --mail-user=toti010@naver.com          

cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load openmpi/4.0.7

# Activate conda environment
source ~/.bashrc
conda activate topo

echo
echo "The following nodes will be used to run this program:"
srun hostname
echo

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_IP=$(hostname -I | awk '{print $1}')

echo "Master address for debugging: $MASTER_ADDR, IP: $MASTER_IP"
#squeue -p gpu | sed -n 's/.*workergpu\([0-9]*\)/\1/p' | sed 's/\([0-9]*\)/'\''\1'\''/g' | paste -sd, - | sed 's/^/[ /;s/$/ ]/'
#squeue -p gpupreempt | sed -n 's/.*workergpu\([0-9]*\)/\1/p' | sed 's/\([0-9]*\)/'\''\1'\''/g' | paste -sd, - | sed 's/^/[ /;s/$/ ]/'

#export CUDA_LAUNCH_BLOCKING=1

srun torchrun \
  --nproc_per_node=4 \
  --nnodes=8 \
  --rdzv_id="gnn_${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:12345" \
  tune.py --layerType SimpleGNN