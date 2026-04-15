## Stochastic Spatial Modeling of Predator–Prey Dynamics in the Lynx–Hare System
In this project, we explore how individual-level movement influences large-scale population dynamics in the classic lynx–hare predator–prey system. We develop a stochastic movement model for lynx and integrate it into a Monte Carlo simulation framework to simulate population dynamics across space and time. We investigate how different movement behaviors and spatial configurations affect system stability, population cycles, and extinction risk.

This work was conducted as part of my bachelor’s thesis at the University of Chicago. While the project is still ongoing, this repository contains all code written up to the point of thesis submission.

### Setting up Environment
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

**References**  
- Original data source: [USFWS database](https://iris.fws.gov/APPS/ServCat/Reference/Profile/165521)  
- CTN_Refuge data: [USFWS database](https://iris.fws.gov/APPS/ServCat/Reference/Profile/165519)  
- Original data cleaning pipeline: [`data_import.r`](https://iris.fws.gov/APPS/ServCat/Reference/Profile/168327)  
- Reference paper: [PNAS, 2024](https://www.pnas.org/doi/10.1073/pnas.2414052121)


### Data Cleaning
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


### State Classification
The state classification pipeline is located in `/code/02_stateClassification/`. The outputs for this pipeline will appear in:

- `/data/processed/stateClassification/` — cleaned CSV files, WMSD files
- `/outputs/classification_diagnostics/` — plots and diagnostic figures

**Note:** Step 1 takes a while to run (10-15 minutes), so I provide the windowed median squared displacement (WMSD) files in `/data/processed/stateClassification/`. By downloading them and putting them in the same location, the pipeline will automatically skip step 1. Otherwise, step 1 will proceed normally and calculate the WMSD values and cache them in the same location. 

To run the pipeline, use the `stateClassification.sh` script. Navigate to `/code/02_stateClassification/` and run the following command: `./stateClassification.sh --config /path/to/config.py`

This command runs the entire pipeline in one go. Each step can also be executed individually if needed. The state classification pipeline consists of three main steps:

1. **01_wmsdCalculation.py**  
   Calculates windowed median squared displacement (WMSD). This script takes long to run, so I provided the pre-calculated cached files which are in       `/data/processed/stateClassification/`. By downloading these files and placing them in the same relative location, the script will skip this step.

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
   

  
### Movement Model
The state classification pipeline is located in `/code/03_movementModel/`. The outputs for this pipeline will appear in:

- `/data/processed/movementModel/` — fit parameters for the movement model
- `/outputs/movement_diagnostics/` — plots and diagnostic figures

To run the pipeline, use the `movementModel.sh` script. Navigate to `/code/03_movementModel/` and run the following command: `./movementModel.sh --config /path/to/config.py`

This command runs the entire pipeline in one go. Each step can also be executed individually if needed. The state classification pipeline consists of three main steps:

1. **01_fittingParameters.py**  
   Saves fitted parameters to a CSV file. Parameters are selected using both quantitative and qualitative criteria. Also includes the fitting workflow (parameter sweeps and comparisons of state-specific MSD means and distributions), computes convex hull area distributions for the stationary state, and calculates the probability of home switching after exploratory movement as a function of distance from home. Plots are saved in `/outputs/movement_diagnostics/01_distributions`.

2. **02_simulationDiagnostics.py**  
   Runs movement simulations and generates diagnostic plots saved in `/outputs/movement_diagnostics/02_diagnostics`. Outputs include:
   - State-specific MSD plots from simulations  
   - Velocity and turning angle distributions by state  
   - Example simulated trajectories
  
### Population Model
The population model simulation code is located in `/code/04_populationModel/`.

The population model was implemented and run on the Midway3 computing cluster provided by the University of Chicago’s Research Computing Center. Batch simulations were executed using a SLURM array job workflow, with the submission script provided in `runSimulation.sbatch`.

**Running the code on Midway3:**
To run the model on Midway3, update the file paths in the SLURM submission script, in particular the location of `fitParameters.csv`, to match the directory structure of your environment. In addition, the command-line flags in the **runSimulation.sbatch** file need to be changed, and the output paths in the **fullSimulation.py** file need to be updated.

**Running the code locally:**
To run the model locally, the file **fullSimulation_local.py** can be executed directly from the command line using: `python fullSimulation_local.py --home /path/to/basedirectory --lambda_12_val <lambda_12_value>`. The resulting .npz file of the time-series will be saved in `outputs/simulation_output/`. While I don't reccomend running the full simulation locally because of the computational cost, the runtime can be reduced by decreasing TMAX in the .py file.
