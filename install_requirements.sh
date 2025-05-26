#!/bin/bash

echo "Installing required Python packages..."
pip install --upgrade pip

# Install core scientific packages
pip install numpy pandas scipy scikit-learn

# Gurobi installation
# Note: You must have Gurobi installed and licensed on your system.
# This just installs the Python bindings
pip install gurobipy
pip install matplotlib seaborn  # For plotting and visual debugging
pip install tqdm                # For progress bars during loops
pip install ipython             # Enhanced interactive shell
pip install jupyterlab          # If you want to run notebooks
pip install joblib              # For caching, parallelism
pip install pyinstrument       # For profiling performance
echo "All required packages installed."
