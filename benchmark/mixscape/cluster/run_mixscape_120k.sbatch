#!/bin/bash

#SBATCH -J mixscape
#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --output=slurm_out/mixscape.%j
#SBATCH --error=slurm_out/mixscape.%j
#SBATCH -t 0-08:00:00
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

source $HOME/.bashrc
conda activate pertpy_env

python mixscape.py --n_cells 120000 --rep 1 --commit ef48ccf
