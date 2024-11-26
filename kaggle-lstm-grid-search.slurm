#!/bin/bash
#SBATCH --job-name=kaggle_lstm_grid_search
#SBATCH --output=results/slurm-%A_%a.out
#SBATCH --error=results/slurm-%A_%a.err
#SBATCH --array=0-36  # Replace with len(param_combinations)
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=48:00:00

module load python/3.11
source venv/bin/activate

python kaggle-lstm-grid-search.py

PARAM_FILE="configs/grid_search.json"
PARAMS=$(jq ".[$SLURM_ARRAY_TASK_ID]" $PARAM_FILE)

python src/train_simpl_lstm.py --config_override '$PARAMS'

deactivate
