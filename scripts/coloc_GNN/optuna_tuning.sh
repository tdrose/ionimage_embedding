#!/bin/bash
#SBATCH -J optuna_tuning                                       # job name
#SBATCH -A alexandr                                            # group to which you belong
#SBATCH -N 1                                                   # number of nodes
#SBATCH -n 8                                                   # number of cores
#SBATCH -p gpu-el8                                             # GPU partition
#SBATCH -C gaming                                              # GPU type                         
#SBATCH --gpus 1                                               # number of GPUs
#SBATCH --ntasks-per-gpu 1                                     # number of tasks per GPU
#SBATCH --cpus-per-gpu 8                                       # number of CPU cores per GPU
#SBATCH --mem-per-gpu 32G                                      # memory pool for each GPU          
#SBATCH -t 0-60:00:00                                          # runtime limit (D-HH:MM:SS)
#SBATCH -o /scratch/trose/slurm_optuna_tuning_.%j.out          # STDOUT
#SBATCH -e /scratch/trose/slurm_optuna_tuning_.%j.err          # STDERR
#SBATCH --mail-type=END,FAIL                                   # notifications for job done & fail
#SBATCH --mail-user=tim.rose@embl.de                           # send-to address

module load CUDA/12.2.0
module load Anaconda3
conda_path=$(which conda)
conda_dir=$(dirname $conda_path)
conda_parent_dir=$(dirname $conda_dir)
conda_sh_path="$conda_parent_dir/etc/profile.d/conda.sh"
source $conda_sh_path
conda activate torch-gpuCUDA12

cd /home/trose/projects/deep_MultiData_mzClustering/scripts/coloc_GNN

python3 -u optuna_tuning.py

