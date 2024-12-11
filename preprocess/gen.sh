#!/bin/sh
#PBS -V                                    
#PBS -v LD_LIBRARY_PATH=$LD_LIBRARY_PATH  
#PBS -q workq
#PBS -N preprocessing
#PBS -l nodes=node11:ppn=36+node12:ppn=36+node13:ppn=36
#PBS -l walltime=100:00:00
#PBS -m abe
#PBS -M toti010@naver.com
#PBS -j oe

cd $PBS_O_WORKDIR
 
module add /appl/modulefiles/intel_compiler_2019 
module add /appl/modulefiles/mpich-3.3.1-intel-ucx 

export PATH=$PATH

echo
echo The following nodes will be used to run this program:
echo                                                        
echo $PBS_NODEFILE > stderr
echo $PBS_NODEFILE > stdout
echo

source ~/.bashrc
conda activate topo >> stderr

mpiexec -n 108 python3 generate_cc.py 1>> stdout 2>> stderr
exit 0


