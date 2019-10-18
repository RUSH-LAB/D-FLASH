#!/bin/bash

#SBATCH --job-name=NotsTest
#SBATCH --nodes=4 --exclusive
#SBATCH --time=00:20:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint=opath

srun ./runme
