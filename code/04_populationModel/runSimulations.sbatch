#!/bin/bash

#SBATCH --job-name=lynx_sim
#SBATCH --output=../BatchOutput/fullSimulation_%A_%a.out
#SBATCH --error=../BatchOutput/fullSimulation_%A_%a.err
#SBATCH --account=pi-jnirody
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --array=0-499
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --time=36:00:00

module load R/4.2.0

source /software/python-miniforge-24.1.2-el8-x86_64/etc/profile.d/conda.sh
conda activate /home/adomenig/.conda/envs/solitary_mammals

path="/path/to/fitParameters.csv"

echo "Running task $SLURM_ARRAY_TASK_ID"

python ./fullSimulation.py \
    --parameters "$path"
