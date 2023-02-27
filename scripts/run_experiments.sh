#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="dvae-eq"

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

# Send yourself an email when the job:
#SBATCH --mail-user=nsranasinghe@student.unimelb.edu.au
#SBATCH --mail-type=BEGIN,FAIL,END

source /usr/local/module/spartan_new.sh
module load anaconda3/2021.11
eval "$(conda shell.bash hook)"
conda activate dvae
#Run program
cd ..
python train.py --data-name eq_structures_2 --data-type EQ --save-interval 100 --save-appendix _DVAE_EQ_20K_300epoch --epochs 300 --lr 1e-4 --model DVAE --bidirectional --nz 56 --batch-size 32
