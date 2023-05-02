#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="dvae-bo-eq"

# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn 
#SBATCH --constraint=[dlg1|dlg2|dlg3]

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=02-00:00:0

# job array
#SBATCH --array=0-0

# Send yourself an email when the job:
#SBATCH --mail-user=nsranasinghe@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

source /usr/local/module/tan_new.sh
module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda activate dvae
#Run program
cd ..
python bayesian_optimization/bo.py \
  --data-name eq_structures_7_w_vals \
  --save-appendix DVAE_EQ_83K_w_vals_200epoch \
  --checkpoint 300 \
  --res-dir="EQ_results/EQ_results_25_${SLURM_ARRAY_TASK_ID}/" \
  --BO-rounds 1 \
  --BO-batch-size 5 \
  --random-as-test \
  --random-as-train \
  --random-baseline \
  --dnum ${SLURM_ARRAY_TASK_ID} \
  --cond
