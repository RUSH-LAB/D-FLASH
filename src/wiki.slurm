#!/bin/bash

#SBATCH --job-name=D-FLASH-WIKI
#SBATCH --nodes=4 --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint="cascadelake&opath"

srun ./runme
