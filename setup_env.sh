#!/bin/bash

# --- Configuration ---
# Set the desired environment name and Python version
ENV_NAME="padim"
PYTHON_VERSION="3.13"

# --- Script ---
echo "Starting environment setup for '$ENV_NAME'..."

# 1. Create the Conda environment
echo "--> Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Check if the environment was created successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment. Aborting."
    exit 1
fi

# 2. Install pip (though it's usually included, this is good practice)
echo "--> Installing pip into '$ENV_NAME'..."
conda run -n $ENV_NAME conda install pip -y

# 3. Install packages from requirements.txt
#    Make sure a 'requirements.txt' file exists in the same directory as this script.
if [ -f "requirements.txt" ]; then
    echo "--> Installing packages from requirements.txt..."
    conda run -n $ENV_NAME pip install -r requirements.txt
else
    echo "Warning: 'requirements.txt' not found. Skipping pip install step."
fi

echo "---"
echo "Setup complete! To activate the environment, run:"
echo "conda activate $ENV_NAME"
echo "---"