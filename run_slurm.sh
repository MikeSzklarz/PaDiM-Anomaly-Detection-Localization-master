#!/bin/bash

# --- SBATCH Directives ---
#SBATCH --job-name=padim               # Job name
#SBATCH --output=logs/slurm/%j.out     # Standard output log
#SBATCH --error=logs/slurm/%j.err      # Standard error log
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=0                        # Memory request (0 = use all available memory)
#SBATCH --time=010:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=waccamaw
#SBATCH --exclusive

# --- Script Body ---

# Exit on any error
set -e

# Print commands and their arguments as they are executed
set -x

# Create log directory if it doesn't exist
# this doesnt work and I dont know why
# I need to create the logs/slurm directory before running sbatch or else it errors
mkdir -p logs/slurm

# --- Default Python script to run (can be overridden with -s / --script) ---
PY_SCRIPT="run_experiment.py"

# --- Parse arguments: allow --script / -s to specify which python script to run.
# Remaining args are passed to the Python script.
ARGS_TO_PY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--script)
      PY_SCRIPT="$2"
      shift 2
      ;;
    --script=*)
      PY_SCRIPT="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [-s|--script SCRIPT.py] [args...]"
      echo "Example: $0 -s run_experiment_cross.py -- --data_path /data --test_class_name foo"
      exit 0
      ;;
    --) # explicit end of options
      shift
      while [[ $# -gt 0 ]]; do
        ARGS_TO_PY+=("$1")
        shift
      done
      ;;
    *)
      ARGS_TO_PY+=("$1")
      shift
      ;;
  esac
done

# --- Environment Setup ---
echo "Loading Conda environment..."
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate padim

echo "Environment loaded."

# --- Execute the Python Script ---
echo "Starting Python script: $PY_SCRIPT"
srun python "$PY_SCRIPT" "${ARGS_TO_PY[@]}"

echo "Job finished successfully."