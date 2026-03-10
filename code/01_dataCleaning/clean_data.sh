#!/bin/bash

###############################################################################
# clean_data.sh
#
# Master script for cleaning and processing lynx trajectory data. 
# This script runs the entire data cleaning pipeline, 
# but each step can also be executed individually if desired.
#
# Steps:
#   1. Clean raw lynx trajectory data (script copied from )
#   2. Align timestamps
#   3. Remove outliers
#   4. Generate basic diagnostic plots
#
# Usage:
#   ./clean_data.sh --config path/to/config.py
#
# Arguments:
#   --config : Path to the Python configuration file. This file must define 
#              a variable "home_directory" that points to the root directory.
#
# Data Source and References:
#   - The lynx trajectory data were obtained from the USFWS IRIS database:
#       https://iris.fws.gov/APPS/ServCat/Reference/Profile/165521

#   - The initial data cleaning steps in 01_lynxCleaning.R closely follow 
#     the original authors' workflow, originally provided as `data_import.r`:
#       https://iris.fws.gov/APPS/ServCat/Reference/Profile/168327

#   - For scientific context and methodology of how the lynx data was collected, 
#     see the publication describing this dataset:
#       https://www.pnas.org/doi/10.1073/pnas.2414052121
###############################################################################

set -e

# parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG_FILE="$2"
      shift 
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# check if config file was provided
if [[ -z "$CONFIG_FILE" ]]; then
  echo "Error: --config path/to/config.py is required"
  exit 1
fi

# get the home directory from config.py using Python
home_dir=$(python3 - <<END
import re
with open("$CONFIG_FILE") as f:
    txt = f.read()
match = re.search(r'home_directory\s*=\s*"(.*)"', txt)
if match:
    print(match.group(1))
END
)

# check if we got it
if [ -z "$home_dir" ]; then
  echo "Error: Could not parse home_directory from $CONFIG_FILE"
  exit 1
fi

echo "Setting home directory to: $home_dir"

# running the scriptst
echo "-------------------------------------------------------------------------------------"
echo "Running 01_lynxCleaning.r"
Rscript "$(dirname "$CONFIG_FILE")/code/01_dataCleaning/01_lynxCleaning.r" "$home_dir"
echo "Completed 01_lynxCleaning.r"
echo "-------------------------------------------------------------------------------------"
echo "Running 02_alignTimes.py"
python "$(dirname "$CONFIG_FILE")/code/01_dataCleaning/02_alignTimes.py" "$home_dir"
echo "Completed 02_alignTimes.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 03_removingOutliers.py"
python "$(dirname "$CONFIG_FILE")/code/01_dataCleaning/03_removingOutliers.py" "$home_dir"
echo "Completed 03_removingOutliers.py"
echo "-------------------------------------------------------------------------------------"
echo "Running 04_diagnostics.py"
python "$(dirname "$CONFIG_FILE")/code/01_dataCleaning/04_diagnotics.py" "$home_dir"
echo "Completed 04_diagnostics.py"
echo "-------------------------------------------------------------------------------------"


