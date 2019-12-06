#!/bin/bash

#SBATCH --job-name=D-FLASH25
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:10:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH --constraint="cascadelake&opath"

srun ./runme
