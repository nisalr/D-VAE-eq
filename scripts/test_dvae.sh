#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="test-dvae-eq"

# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn 
##SBATCH --constraint=[dlg1|dlg2|dlg3]
##SBATCH --partition=feit-gpu-a100
##SBATCH --qos=feit

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=07-00:00:0

# Send yourself an email when the job:
#SBATCH --mail-user=nsranasinghe@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

source /usr/local/module/spartan_new.sh
module load anaconda3/2021.11
module load gcc/9.2.0
eval "$(conda shell.bash hook)"
conda activate dvae_2
#Run program
cd ..
echo 'running program'
#python train.py --data-name eq_structures_2 --data-type EQ --save-interval 25 --save-appendix _DVAE_EQ_20K_100epoch --epochs 100 --lr 1e-4 --model DVAE --bidirectional --nz 56 --batch-size 32 --nvt 5 --only-test --continue-from 100

#python train.py --data-name eq_structures_8_w_vals --data-type EQ --save-interval 25 --save-appendix _DVAE_EQ_120K_w_vals_100epoch --epochs 100 --lr 1e-4 --model DVAE --bidirectional --nz 56 --batch-size 32 --nvt 5 --cond --only-test --continue-from 100 --cond-size 9

python train.py --data-name eq_structures_22_nesym --data-type EQ --save-interval 25 --save-appendix _DVAE_EQ_20K_nesym_100epoch --epochs 100 --lr 1e-4 --model DVAE --bidirectional --nz 56 --batch-size 32 --nvt 5 --cond --cond-size 10 --continue-from 50 --only-test
