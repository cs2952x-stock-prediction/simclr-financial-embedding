#!/bin/bash
#SBATCH -J kaggle_lstm_grid_search
#SBATCH -o results/slurm-%A_%a.out
#SBATCH -e results/slurm-%A_%a.err
#SBATCH --array=0-9
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

# make sure the right python resources are loaded
module load python/3.11
source venv/bin/activate

# NOTE: make sure to run 'python src/generate_feature_ablations.py'
# beforehand to generate the grid-search parameters

PARAM_FILE="configs/feature_ablations.json"

# Explanation of the command below:
#
# jq - "json query"
# We use this to extract the json representation of our grid parameters
#
# jq takes 2 arguments:
# 1) A query term:
#    Using ".[idx]" will extract index 'idx' of a json array.
#    Using ".[$SLURM_ARRAY_TASK_ID]" will index into the json list/array
#    of parameters based on the current task ID.
# 2) The file in which contains the json
PARAMS=$(jq ".[$SLURM_ARRAY_TASK_ID]" $PARAM_FILE)

# we pass the grid parameters we extracted using 'config_everride'
# this shadows/overrides the default config values with the new ones
python src/train_simple_lstm.py --config_override "$PARAMS"

deactivate
