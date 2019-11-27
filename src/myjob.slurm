#!/bin/bash

#SBATCH --job-name=NotsTest
#SBATCH --nodes=3 --exclusive 
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun ./runme
