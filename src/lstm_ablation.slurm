#!/bin/bash
#SBATCH -J kaggle_lstm_ablation
#SBATCH -o ablation_results/slurm-%A_%a.out
#SBATCH -e ablation_results/slurm-%A_%a.err
#SBATCH --array=0-8
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH -p gpu

# What these options do:
# J: sets the job name
# o: the file where standard out will be logged
# e: the file where standard error will be logged
# array: creates an array of tasks with IDs start_idx-end_idx
# 			 each of these tasks will run the bash script below
# 			 but with $SLURM_ARRAY_TASK_ID set to the task id
# c: the number of cpu cores per task
# mem: the amount of memory to allocate each task
# time: the wall-time limit of each task
# p: the partition on which to run the job
#    (different partitions have different resources)

# Load Python and activate virtual environment
module load python/3.11
source venv/bin/activate

# Define the ablations array
ABLATIONS=(
    ""
    "week_sin week_cos"
    "month_sin month_cos"
    "quarter_sin quarter_cos"
    "year_sin year_cos"
    "close_ema20"
    "close_ema50"
    "close_sma20"
    "close_sma50"
)

# Access the current ablation based on SLURM_ARRAY_TASK_ID
CURRENT_ABLATION="${ABLATIONS[$SLURM_ARRAY_TASK_ID]}"

# Run the Python script with the selected ablation
python src/train_simple_lstm.py -x "open min max close volume ${CURRENT_ABLATION}" --name "ablation_$SLURM_ARRAY_TASK_ID"

# Deactivate virtual environment
deactivate
