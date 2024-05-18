#!/bin/bash

#SBATCH --job-name=run_gobeavers
#SBATCH --output=/central/groups/CS156b/2024/gobeavers/cs156b/%j.out
#SBATCH --error=/central/groups/CS156b/2024/gobeavers/cs156b/%j.err
#SBATCH -A CS156b
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres gpu:1
#SBATCH --mail-user=jdowns@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --partition=gpu

source /home/jdowns/.bashrc
module load cuda/10.2

conda activate gb

python feature_resnet.py "No Finding"
