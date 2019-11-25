#!/bin/bash

#SBATCH --job-name=NotsTest
#SBATCH --nodes=1 --exclusive 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun ./runme
