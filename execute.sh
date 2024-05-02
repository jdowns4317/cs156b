#!/bin/bash

#SBATCH --job-name=run_gobeavers
#SBATCH --output=/central/groups/CS156b/2024/gobeavers/cs156b/%j.out
#SBATCH --error=/central/groups/CS156b/2024/gobeavers/cs156b/%j.err
#SBATCH -A CS156b
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres gpu:4
#SBATCH --mail-user=jdowns@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/jdowns/.bashrc

conda activate gb156

python model.py
