#!/bin/bash
#SBATCH --job-name=run                         
#SBATCH --output=stdout                    
#SBATCH --error=stderr                     
#SBATCH --nodes=2                          
#SBATCH --ntasks=2                          
#SBATCH --cpus-per-task=48                     
#SBATCH --time=120:00:00                       
#SBATCH --partition=gpu
#SBATCH --gpus=a100-sxm4-40gb:8       
#SBATCH --mem=0
#SBATCH --mail-type=ALL                        
#SBATCH --mail-user=toti010@naver.com         

module load openmpi/4.0.7
source ~/.bashrc
conda activate topo

types=("Quijote")  
layerTypes=("GNN" "TetraTNN" "ClusterTNN" "TNN" "All")

MASTER_IP=$(hostname -I | awk '{print $1}')

for layer in "${layerTypes[@]}"; do
  for TYPE in "${types[@]}"; do
    export TYPE
    echo "Running TYPE=${TYPE}, layerType=${layer}"

    srun torchrun \
      --nproc_per_node=4 \
      --nnodes=2 \
      --rdzv_id="gnn_${SLURM_JOB_ID}_${TYPE}_${layer}_${feat}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_IP}:12345" \
      tune.py --layerType "$layer"
  done
done


