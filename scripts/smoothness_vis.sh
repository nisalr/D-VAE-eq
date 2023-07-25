#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="bo-smooth"

# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn 
#SBATCH --constraint=[dlg1|dlg2|dlg3]

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=02-00:00:0

# Send yourself an email when the job:
#SBATCH --mail-user=nsranasinghe@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

source /usr/local/module/tan_new.sh
module load anaconda3/2021.11
module load gcc/9.2.0
eval "$(conda shell.bash hook)"
conda activate dvae_2
#Run program
cd ..

python bayesian_optimization/bo.py \
  --data-name eq_structures_23_nesym \
  --save-appendix DVAE_EQ_120K_nesym_100epoch \
  --checkpoint 100 \
  --res-dir="EQ_results_vis_21_nesym_pred/" \
  --BO-rounds 4 \
  --BO-batch-size 200 \
  --random-as-test \
  --random-as-train \
  --random-baseline \
  --vis-2d \
  --dnum 0 \
  --cond \
  --cond-size 10

#python bayesian_optimization/bo.py \
#  --data-name eq_structures_3 \
#  --save-appendix DVAE_EQ_120K_200epoch \
#  --checkpoint 200 \
#  --res-dir="EQ_results_vis_10_no_cond/" \
#  --BO-rounds 4 \
#  --BO-batch-size 200 \
#  --random-as-test \
#  --random-as-train \
#  --random-baseline \
#  --vis-2d \
#  --dnum 0 \

#python bayesian_optimization/bo.py \
#  --data-name eq_structures_8_w_vals \
#  --save-appendix DVAE_EQ_120K_w_vals_200epoch \
#  --checkpoint 200 \
#  --res-dir="EQ_results_vis_11_w_vals/" \
#  --BO-rounds 4 \
#  --BO-batch-size 200 \
#  --random-as-test \
#  --random-as-train \
#  --random-baseline \
#  --vis-2d \
#  --dnum 0 \
#  --cond \
#  --cond-size 9
