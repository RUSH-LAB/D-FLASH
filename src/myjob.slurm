#!/bin/bash

#SBATCH --job-name=NotsTest
#SBATCH --ntasks=4 --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun ./runme
