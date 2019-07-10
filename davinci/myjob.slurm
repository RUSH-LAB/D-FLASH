#!/bin/bash
#SBATCH --job-name=FLASH_Test1.0
#SBATCH --account=commons
#SBATCH --partition=interactive
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=00:10:00
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=ALL
 
echo "My job ran on:"
echo $SLURM_NODELIST

srun $HOME/src/runme
