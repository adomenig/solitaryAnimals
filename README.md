#
solitaryAnimals
## Setting up Environment
Before running any of the scripts, we need to set up the environment first. To do so, follow these steps below: 
1. Download the lynx trajectory dataset
   - File: `Complete_20231231`  
   - Source: [USFWS database](https://iris.fws.gov/APPS/ServCat/Reference/Profile/165521)  
   - Place it in: `/data/Complete_20231231/`

2. Download the CTN refuge reference data  
   - File: `ctn_refuge.csv`  
   - Source: [USFWS database](https://iris.fws.gov/APPS/ServCat/Reference/Profile/165519)  
   - Place it in: `/data/Complete_20231231/ctn_refuge/`

3. Update the configuration file
   - Edit `config.py` to reflect the root directory of your project.

**Required Packages**: You will need an environment with the following packages installed to run the scripts:

- Python packages
  - sys
  - matplotlib
  - numpy
  - pandas
  - pathlib
  - concurrent.futures
  - tqdm
  - seaborn
  - mpl_toolkits.basemap
  - haversine

- R packages
  - tidyverse
  - lubridate

**References**  
- Original data source: [USFWS database](https://iris.fws.gov/APPS/ServCat/Reference/Profile/165521)  
- CTN_Refuge data: [USFWS database](https://iris.fws.gov/APPS/ServCat/Reference/Profile/165519)  
- Original data cleaning pipeline: [`data_import.r`](https://iris.fws.gov/APPS/ServCat/Reference/Profile/168327)  
- Reference paper: [PNAS, 2024](https://www.pnas.org/doi/10.1073/pnas.2414052121)


## Data Cleaning
The data cleaning pipeline is located in `/code/01_dataCleaning/`. Before running it, make sure you followed the setup steps above. The outputs for this pipeline will appear in:

- `/data/processed/dataCleaning/` — cleaned CSV files  
- `/outputs/data_diagnostics/` — plots and diagnostic figures

To run the pipeline, use the `cleanData.sh` script. Navigate to `/code/01_dataCleaning/` and run the following command: `./cleanData.sh --config /path/to/config.py`

This command runs the entire pipeline in one go. Each step can also be executed individually if needed. The data cleaning pipeline for the lynx trajectory dataset consists of four main steps:

1. **01_lynxCleaning.R**  
   This script was sourced from the original authors who collected the lynx trajectory data. It handles the initial import, formatting, and standardization of the raw data to prepare it for analysis. The workflow closely follows the original `data_import.r` script from the USFWS database with some minor tweaks in formatting.

2. **02_alignTimes.py**  
   Aligns timestamps across all lynx trajectories to ensure we have consistent 4-hour timesteps. This step corrects for differences in recording intervals and prepares the dataset for downstream analyses such as MSD calculations.

3. **03_removingOutliers.py**  
   Identifies and removes outlier locations that are likely errors or biologically implausible. It also corrects minor inconsistencies in formatting.
   
4. **04_diagnostics.py**  
   Generates diagnostic plots and summaries for the cleaned dataset. This includes:
   - Distributions of lynx by sex and age
   - Daily counts of tracked lynx
   - Trajectory plots over a basemap
   - Mean squared displacement (MSD) analyses
   - Instantaneous velocity distributions for selected individuals


## State Classification
The state classification pipeline is located in `/code/02_stateClassification/`. The outputs for this pipeline will appear in:

- `/data/processed/stateClassification/` — cleaned CSV files  
- `/outputs/classification_diagnostics/` — plots and diagnostic figures

To run the pipeline, use the `stateClassification.sh` script. Navigate to `/code/02_stateClassification/` and run the following command: `./stateClassification.sh --config /path/to/config.py`

This command runs the entire pipeline in one go. Each step can also be executed individually if needed. The state classification pipeline consists of three main steps:

1. **01_wmsdCalculation.py**  
   Calculates windowed median squared displacement (WMSD). This script takes long to run, so I provided the pre-calculated cached files which are in       `/data/processed/stateClassification/`. By downloading these files and keeping them in the same relative location, the script will skip this step.

2. **02(a + b)_stateClassification.py**  
   a) Assigns states based on the WMSD values calculated in the previous step.
   
   b) Generates diagnostic plots for the assigned states. This includes:
      - Plots of the MSDs split by state
      - Plots of the velocity and turning angle distributions split by state
      - Example trajectories colored by assigned states

4. **03(a + b)_splittingLoops.py**  
   a) Identifies when state 2 causes a territory switch or when the lynx loops back to the same territory. If looping, assigns state 3 to the inbound/returning loop.
   
   b) Generates diagnostic plots for the assigned states. This includes:
      - Plots of the MSDs split by state
      - Plots of the velocity and turning angle distributions split by state
      - Example trajectories colored by assigned states
   

  

