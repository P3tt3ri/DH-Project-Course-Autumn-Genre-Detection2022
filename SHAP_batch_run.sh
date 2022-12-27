#!/bin/bash
#SBATCH --account=project_2006463
#SBATCH --job-name=PetteriSHAPtest
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH -o SHAP_batch.out
#SBATCH -e SHAP_batch.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=07:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1

module load python-data 

srun python3 SHAP_batch.py
