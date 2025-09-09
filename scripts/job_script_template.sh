#!/bin/bash

# SPDX-FileCopyrightText: 2025 GFZ Helmholtz Centre for Geosciences
# SPDX-FileContributor: Bernhard Haas
#
# SPDX-License-Identifier: Apache 2.0

#SBATCH --job-name=data_processing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=4
#SBATCH --output=output_%j.log
#SBATCH --reservation=download

# Get the start and end times from the command line arguments
START_TIME=$1
END_TIME=$2

echo "Processing data from $START_TIME to $END_TIME"

# Replace 'your_program.py' with the path to your data processing script.
python your_progam.py "$START_TIME" "$END_TIME" "path_to_irbem/libirbem.so"