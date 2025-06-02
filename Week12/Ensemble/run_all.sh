#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Path to your virtual environment
VENV_PATH="/Users/justincase/PycharmProjects/DowJones_Forecasts/.venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

echo "Virtual environment activated."

# Change to the directory where your scripts are
cd "$(dirname "$0")"
echo "Current working directory: $(pwd)"


# Run your Python scripts
echo "Running script1.py..."
python Lasso_Model.py

echo "Running script2.py..."
python Ridge_Model.py

echo "Running script3.py..."
python Elastic_Net_Model.py

echo "Running script4.py..."
python PCA_Model_1.py

echo "Running script5.py..."
python PCA_Model_2.py

echo "Running script6.py..."
python PCA_Model_3.py

echo "Running script7.py..."
python PCA_Model_4.py

echo "Running script8.py..."
python PCA_Model_5.py

echo "Running script9.py..."
python PCA_Model_6.py

echo "Running script10.py..."
python PCA_Model_7.py

echo "Running script11.py..."
python PCA_Model_8.py

# Optional: deactivate the environment
deactivate