#!/bin/bash

#SBATCH --job-name=NotsTest
#SBATCH --ntasks=10 --exclusive
#SBATCH --time=00:20:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun ./runme
