#!/bin/bash

# --- SBATCH Directives ---
#SBATCH --job-name=anomalib_training   # Job name
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
mkdir -p logs/slurm

# Capture all command-line arguments passed to this script
ARGS="$@"

# --- Environment Setup ---
echo "Loading Conda environment..."
source /mnt/cidstore1/software/debian12/anaconda3/etc/profile.d/conda.sh
conda activate padim

echo "Environment loaded."

# --- Execute the Python Script ---
echo "Starting Python script..."

# Use srun to execute the python script, passing all captured arguments
srun python run_experiment.py ${ARGS}

echo "Job finished successfully."