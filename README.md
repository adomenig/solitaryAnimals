# solitaryAnimals
## Setting up
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

- `/data/processed/` — cleaned CSV files  
- `/outputs/data_diagnostics/` — plots and diagnostic figures

To run the pipeline, use the `clean_data.sh` script. Navigate to `/code/01_dataCleaning/` and run the following command: `./clean_data.sh --config /path/to/config.py`

This command runs the entire pipeline in one go. Each step can also be executed individually if needed. The data cleaning pipeline for the lynx trajectory dataset consists of four main steps:

1. **01_lynxCleaning.R**  
   This script was sourced from the original authors who collected the lynx trajectory data. It handles the initial import, formatting, and standardization of the raw data to prepare it for analysis. The workflow closely follows the original `data_import.r` script from the USFWS database with some minor tweaks.

2. **02_alignTimes.py**  
   Aligns timestamps across all lynx trajectories to ensure we have consistent 4-hour timesteps. This step corrects for differences in recording intervals and prepares the dataset for downstream analyses such as MSD calculations.

3. **03_removingOutliers.py**  
   Identifies and removes outlier locations that are likely errors or biologically implausible. It also corrects minor inconsistencies, such as duplicate trajectories (e.g., WSM025 and WSM045).
   
4. **04_diagnostics.py**  
   Generates diagnostic plots and summaries for the cleaned dataset. This includes:
   - Distributions of lynx by sex and age
   - Daily counts of tracked lynx
   - Trajectory plots over a basemap
   - Mean squared displacement (MSD) analyses
   - Instantaneous velocity distributions for selected individuals
  

