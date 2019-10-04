#!/bin/bash

#SBATCH --job-name=NotsTest
#SBATCH --ntasks=2 --exclusive
#SBATCH --time=00:10:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun ./runme
