import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
import os
import time
import h5py
import argparse
from scipy.stats import expon
import sys

# Command-line arguments
parser = argparse.ArgumentParser(description="Run lynx-hare simulations with different initial conditions")
parser.add_argument("--initial", type=str, choices=["EW", "uniform", "gaussian"], default="uniform",
                    help="Type of initial conditions for lynx positions")
parser.add_argument("--parameters", type=str, required=True, help="File path to fitParameters.csv")

args = parser.parse_args()
initial_type = args.initial
parameter_path = Path(args.parameters)

print(f"Using initial conditions: {initial_type}")

########################## PARAMETER INITIALIZATION START #######################
# INITIALIZING PARAMETERS
L_w, L_h = 1000, 1000 # height and width of our field. This is fit so 1 unit = 1 km. 
T_MAX = 200 # maximum time of simulation. This is fit so 1 time unit = 1 year
DT = 4.0 / (24*365.25) # our time step is 4 hours. 
STEPS = int(T_MAX / DT) # the number of steps we need to take calculated by dividing the nr of years we have by our 4-hour time step.

# Hare Parameters
beta0 = 2.0 # Hare birth rate (they will give between 1-3 times a year on average)
sigma = 1/3 # Hare death rate  (they will die from natural causes (not predation-related) every 3 years on average)
D_B = 50 # diffusion rate (they will go to one of their neighboring fields around 50 times a year on average) 

# Lynx parameters
alpha0 = 1.0 # Lynx birth rate (lynx will give birth once a year on average)
delta = 1/15 # Lynx death rate (lynx will die every 15 years from natural causes on average if hares are abundant)
mu = 365.25 / 2.5 # Predation rate (with abundant hares available, they will on average eat a hare every 2-3 days)

# Initial concentrations of lynx and hare. These values will stabilize after the first 1-2 cycles.
initial_hare = 10
initial_lynx = 0.01

params_state1 = {}
params_state2 = {}
params_state3 = {}
lambda_12 = 0
territory_scale = 0
beta0_logistic = 0
beta1_logistic = 0

if parameter_path.exists():
    params_df = pd.read_csv(parameter_path)
    # extract state1 parameters 
    state1_params = params_df[params_df['parameter'].str.startswith("state1_")]
    state1_params = {row['parameter'].replace("state1_", ""): row['value'] for _, row in state1_params.iterrows()}

    # extract state2 parameters
    state2_params = params_df[params_df['parameter'].str.startswith("state2_")]
    state2_params = {row['parameter'].replace("state2_", ""): row['value'] for _, row in state2_params.iterrows()}

    # extract state3 parameters
    state3_params = params_df[params_df['parameter'].str.startswith("state3_")]
    state3_params = {row['parameter'].replace("state3_", ""): row['value'] for _, row in state3_params.iterrows()}

    # extract switching rates
    #lambda_12 = float(params_df[params_df['parameter'] == "lambda_12"]['value'].values[0])
    lambda_21 = float(params_df[params_df['parameter'] == "lambda_21"]['value'].values[0])

    territory_size_row = params_df[params_df['parameter'].str.contains("territory_size_distribution")]
    territory_scale = float(territory_size_row['value'].values[0])

    # extract loop distance scale (exponential fit)
    beta0_logistic = float(params_df.loc[params_df['parameter'] == "state2_logistic_beta0", 'value'].values[0])
    beta1_logistic = float(params_df.loc[params_df['parameter'] == "state2_logistic_beta1", 'value'].values[0])
else: 
    print(f"File with saved parameters missing. Make sure your input is corect.")
    sys.exit(1)

STATE1, STATE2 = 1, 2
# lambda_12 values we are testing
lambda_12_values = [0.0001, 0.00025, 0.0005, 0.001, 0.0015]
n_runs = 9  # number of repeats per scenario
snapshot_interval_steps = 546 # roughly every 3 months
########################## PARAMETER INITIALIZATION END #######################




############################ HELPER FUNCTIONS FOR SIMULATION START ##################
# RATE FUNCTIONS
def delta_of_B(B):
    """
    Predator death rate as a function of local prey density. This is where the main 
    dependance in predator-prey dynamics comes from. This is a hill function with the 
    idea that low prey densities means higher predator death rates. 
    """
    del_max = 0.5
    B_half = initial_hare * 5
    h = 2
    return delta + (del_max - delta) / (1 + (B / B_half) ** h)

# Number of offspring lynx
def k_of_B(B_local):
    """
    Number of offspring per lynx as a function of local prey density. This is a 
    sigmoid step function the n_births approaches 0 at low prey and k_max at high prey. 
    """
    k_min, k_max = 0, 8
    B_half, h = initial_hare * 5, 0.2  # midpoint and steepness
    sigmoid = (k_max) / (1 + np.exp(-h * (B_local - B_half)))
    return np.clip(sigmoid, k_min, k_max).astype(int)


def alpha_func(territory_size, alpha_state1, alpha_state3, dist_to_home):
    """
    Distance-dependent attraction strength toward the home location.

    This function interpolates between two behavioral regimes:
    a stationary-state attraction strength (alpha_state1) and a
    return-movement attraction strength (alpha_state3), as a function
    of the distance from the current position to the home location. The 
    transition between regimes is modeled using a sigmoid centered
    at the territory size. We clip to avoid overflow errors but it's functionally
    the same. 

    Returns
        - Distance-dependent attraction strength alpha.
    """
    steepness = 10
    x = steepness * (dist_to_home - territory_size)
    x = np.clip(x, -50, 50)
    sigmoid = 1 / (1 + np.exp(-x))
    return alpha_state1 + (alpha_state3 - alpha_state1) * sigmoid

def prob_switch_to_3(dist, beta0, beta1):
    """
    Probability of switching from exploratory to territorial state
    as a function of distance from the home range.

    This function evaluates the logistic model fit to empirical state
    transition data, where the probability of initiating a new
    territory increases with distance from the current home location.

    Returns
        - Probability of switching to a new territory (state 3).
    """
    return 1 / (1 + np.exp(-(beta0 + beta1 * dist)))

# compute neighborhood "densities"
def compute_density(B):
    return (
        B
        + np.roll(B, 1, axis=0)
        + np.roll(B, -1, axis=0)
        + np.roll(B, 1, axis=1)
        + np.roll(B, -1, axis=1)
    )
############################ HELPER FUNCTIONS FOR SIMULATION END ##################





####################### MOVEMENT SIMULATION START #################################
# LYNX MOVEMENT
def move_lynx(pos, state, params, B_density, lambda_12, dt=4.0):
    """
    This function is the core logic of the lynx movement. Unlike the MC simulation, all movement was
    fit at dt=4, so therefore we use 4 to multiply the rates and values in this part. The movement consists of two
    states: Stationary and Exploratory. The stationary movement consists of diffusion with a pull towards home
    whereas the exploratory movement is diffusive with an additional velocity term. 
    """
    N = len(pos)

    if N == 0:
        return pos, state, params

    # STATE1: attraction + diffusion + switching
    idx1 = state == STATE1
    n1 = idx1.sum()
    if n1 > 0:
        pos1 = pos[idx1]
        home1 = params["home"][idx1]
        diff = pos1 - home1
        r = np.linalg.norm(diff, axis=1, keepdims=True)
        r[r == 0] = 1  # avoid division by zero
        
        # calculating the current alpha
        dist_to_home = np.linalg.norm(diff, axis=1)
        alpha = alpha_func(params["territory"][idx1], state1_params["alpha"], state3_params["alpha"], dist_to_home)
        alpha = alpha[:, None] # for dimensional matching
        pos1 -= alpha * diff / r * dt
        pos1 += np.sqrt(2 * params["D1"][idx1][:, None] * dt) * np.random.randn(n1, 2)

        switch_mask = np.random.rand(n1) < lambda_12

        idx_switch = np.where(idx1)[0][switch_mask]
        if len(idx_switch) > 0:
            state[idx_switch] = STATE2
            params["v"][idx_switch] = np.random.uniform(state2_params["v_lower"], state2_params["v_higher"], size=len(idx_switch))
            params["D2"][idx_switch] = np.random.uniform(state2_params["D2_lower"], state2_params["D2_higher"], size=len(idx_switch))
            params["Dtheta"][idx_switch] = np.random.uniform(state2_params["Dtheta_lower"], state2_params["Dtheta_higher"], size=len(idx_switch))
            params["theta"][idx_switch] = np.random.uniform(0, 2*np.pi, size=len(idx_switch))
        pos[idx1] = pos1

    # STATE2: persistent motion + diffusion + switching
    idx2 = state == STATE2
    n2 = idx2.sum()
    if n2 > 0:
        theta2 = params["theta"][idx2]
        v2 = params["v"][idx2]
        D2 = params["D2"][idx2]
        Dtheta = params["Dtheta"][idx2]
        theta2 += np.sqrt(2 * Dtheta * dt) * np.random.randn(n2)
        dx = (v2[:, None] * np.column_stack([np.cos(theta2), np.sin(theta2)])) * dt
        dx += np.sqrt(2 * D2[:, None] * dt) * np.random.randn(n2, 2)
        pos2 = pos[idx2] + dx

        switch_mask = np.random.rand(n2) < lambda_21
        idx_switch = np.where(idx2)[0][switch_mask]

        if len(idx_switch) > 0:
            # switch state
            state[idx_switch] = STATE1

            # resample STATE1 diffusion
            params["D1"][idx_switch] = np.random.uniform(state1_params["D1_lower"], state1_params["D1_higher"], size=len(idx_switch))

            # decide whether to update home
            if len(idx_switch) > 0:
                dist_to_home = np.linalg.norm(pos[idx_switch] - params["home"][idx_switch],axis=1)
                p3 = prob_switch_to_3(dist_to_home, beta0_logistic, beta1_logistic)

                update_mask = np.random.rand(len(idx_switch)) > p3
                idx_update = idx_switch[update_mask]

                if len(idx_update) > 0:
                    params["home"][idx_update] = pos[idx_update].copy()
                    params["territory"][idx_update] = np.sqrt(expon.rvs(scale=territory_scale, size=len(idx_update)) / np.pi)
        # always update motion
        params["theta"][idx2] = theta2
        pos[idx2] = pos2

    xmax = L_w - 1
    ymax = L_h - 1

    # X boundaries
    over_x = pos[:, 0] > xmax
    under_x = pos[:, 0] < 0

    if np.any(over_x):
        pos[over_x, 0] = xmax
        params["theta"][over_x] = np.random.uniform(0, 2*np.pi, size=over_x.sum())

    if np.any(under_x):
        pos[under_x, 0] = 0
        params["theta"][under_x] = np.random.uniform(0, 2*np.pi, size=under_x.sum())

    # Y boundaries
    over_y = pos[:, 1] > ymax
    under_y = pos[:, 1] < 0

    if np.any(over_y):
        pos[over_y, 1] = ymax
        params["theta"][over_y] = np.random.uniform(0, 2*np.pi, size=over_y.sum())

    if np.any(under_y):
        pos[under_y, 1] = 0
        params["theta"][under_y] = np.random.uniform(0, 2*np.pi, size=under_y.sum())

    return pos, state, params
####################### MOVEMENT SIMULATION END #################################





####################### SIMULATION REACTIONS START #################################
# REACTION 1: HARE BIRTH B -> B + B
def do_hare_birth(B):
    birth_counts = np.random.poisson(beta0 * B * DT)
    total_births = birth_counts.sum()
    if total_births > 0:
        parent_i, parent_j = np.nonzero(birth_counts)
        repeats = birth_counts[parent_i, parent_j]
        parents_i = np.repeat(parent_i, repeats)
        parents_j = np.repeat(parent_j, repeats)

        directions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        choices = np.random.randint(0, 4, size=len(parents_i))
        di_dj = directions[choices]

        offspring_i = np.clip(parents_i + di_dj[:,0], 0, L_h-1)
        offspring_j = np.clip(parents_j + di_dj[:,1], 0, L_w-1)

        np.add.at(B, (offspring_i, offspring_j), 1)
    return B

# REACTION 2: HARE DEATH B -> NULL
def do_hare_death(B):
    B -= np.random.poisson(sigma * B * DT)
    return np.maximum(B, 0)

# REACTION 3: HARE DIFFUSION B -> MOVEMENT TO A RANDOM NEIGHBOR
def do_hare_diffusion(B):
    if D_B <= 0 or B.sum() == 0:
        return B

    move_prob = 1 - np.exp(-D_B * DT)
    hare_coords = np.argwhere(B > 0)
    n_hare = len(hare_coords)
    if n_hare == 0:
        return B

    move_mask = np.random.rand(n_hare) < move_prob
    moving_hares = hare_coords[move_mask]
    if len(moving_hares) > 0:
        directions = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        choices = np.random.randint(0,4,len(moving_hares))
        displacements = directions[choices]
        new_positions = moving_hares + displacements
        new_positions[:,0] = np.clip(new_positions[:,0], 0, L_h-1)
        new_positions[:,1] = np.clip(new_positions[:,1], 0, L_w-1)
        old_idx = np.ravel_multi_index((moving_hares[:,0], moving_hares[:,1]), dims=B.shape)
        new_idx = np.ravel_multi_index((new_positions[:,0], new_positions[:,1]), dims=B.shape)
        B_ravel = B.ravel()
        np.add.at(B_ravel, old_idx, -1)
        np.add.at(B_ravel, new_idx, 1)
        B = np.maximum(B_ravel.reshape(B.shape), 0)
    return B

# REACTION 4: LYNX DEATH A -> NULL
def do_lynx_death(pos, state, params, B_density):
    if len(pos) == 0:
        return pos, state, params

    i = pos[:, 1].astype(int)
    j = pos[:, 0].astype(int)

    death_rate = delta_of_B(B_density[i, j])  # lynx death rate depends on local prey
    death_events = np.random.poisson(death_rate * DT)

    alive_mask = death_events == 0
    pos = pos[alive_mask]
    state = state[alive_mask]
    for key in params:
        params[key] = params[key][alive_mask]

    return pos, state, params

# REACTION 5: LYNX BIRTH A -> A + A
def do_lynx_birth(pos, state, params, B_density):
    if len(pos) == 0:
        return pos, state, params

    # only lynx in state 1 can give birth
    state1_mask = state == STATE1
    birth_events = np.zeros(len(pos), dtype=int)
    birth_events[state1_mask] = np.random.poisson(alpha0 * DT, size=state1_mask.sum())

    parents_mask = birth_events > 0


    if not np.any(parents_mask):
        return pos, state, params

    parents = pos[parents_mask]
    i = parents[:, 1].astype(int)
    j = parents[:, 0].astype(int)
    B_local = B_density[i, j]

    offspring_counts = k_of_B(B_local)  # number of offspring per parent
    offspring_counts *= birth_events[parents_mask]  # account for multiple birth events

    total_offspring = offspring_counts.sum()
    if total_offspring == 0:
        return pos, state, params

    # Offspring appear at parent's location
    new_pos = np.repeat(parents, offspring_counts, axis=0)
    new_pos[:, 0] = np.clip(new_pos[:, 0], 0, L_w - 1)
    new_pos[:, 1] = np.clip(new_pos[:, 1], 0, L_h - 1)

    # Update positions and states
    pos = np.vstack([pos, new_pos])
    state = np.concatenate([state, np.ones(len(new_pos), dtype=int)])
    params["home"] = np.vstack([params["home"], new_pos])
    params["D1"] = np.concatenate([params["D1"], np.random.uniform(state1_params["D1_lower"], state1_params["D1_higher"], len(new_pos))])
    new_territories = np.sqrt(expon.rvs(scale=territory_scale, size=len(new_pos)) / np.pi)
    params["territory"] = np.concatenate([params["territory"], new_territories])
    for key in ["v", "D2", "theta", "Dtheta"]:
        params[key] = np.concatenate([params[key], np.zeros(len(new_pos))])

    return pos, state, params

# REACTION 6: PREDATION A + B -> A
def do_predation(B, pos, mu):
    if len(pos) == 0 or B.sum() == 0:
        return B

    i = pos[:, 1].astype(int)
    j = pos[:, 0].astype(int)

    B_local = B[i, j]
    expected_pred = mu * B_local * DT
    num_eaten = np.random.poisson(expected_pred)

    total_eaten = np.zeros_like(B)
    np.add.at(total_eaten, (i, j), num_eaten)
    total_eaten = np.minimum(total_eaten, B)

    B -= total_eaten
    return B

# REACTION 7: LYNX MOVEMENT A -> MOVE USING STOCHASTIC MOVEMENT MODEL
def lynx_movement_reaction(pos, state, params, B_density, lambda_12):
    pos, state, params = move_lynx(pos, state, params, B_density, lambda_12)
    return pos, state, params
####################### SIMULATION REACTIONS END #################################





###################################### MAIN SIMULATION START ####################################################
# SIMULATION
def simulate(output_dir, lambda_12_val, run):
    lambda_12 = lambda_12_val
    B = np.full((L_h, L_w), initial_hare, dtype=int)

    N = int(L_h * L_w * initial_lynx)

    pos = np.zeros((N, 2))
    state = np.ones(N, dtype=int)

    if initial_type == "uniform":
        # uniform across entire domain
        pos[:, 0] = np.random.uniform(0, L_w-1, N)
        pos[:, 1] = np.random.uniform(0, L_h-1, N)

    elif initial_type == "EW":
        # right-biased distribution
        pos[:, 0] = np.random.uniform(L_w*0.8, L_w-1, N)
        pos[:, 1] = np.random.uniform(0, L_h-1, N)

    elif initial_type == "gaussian":
        # Three random blobs
        n_blobs = 3
        blob_size = N // n_blobs
        std_dev = min(L_w, L_h) * 0.05  # spread of each blob
        margin = 3 * std_dev  # distance from edges to avoid clipping


        for i in range(n_blobs):
            # random center
            center_x = np.random.uniform(margin, L_w - margin)
            center_y = np.random.uniform(margin, L_h - margin)
            
            start = i * blob_size
            end = (i + 1) * blob_size if i < n_blobs - 1 else N
            
            # sample from Gaussian around the center
            pos[start:end, 0] = np.random.normal(center_x, std_dev, end-start)
            pos[start:end, 1] = np.random.normal(center_y, std_dev, end-start)

    params = {
        "home": pos.copy(),
        "D1": np.random.uniform(state1_params["D1_lower"], state1_params["D1_higher"], N),
        "v": np.zeros(N),
        "D2": np.zeros(N),
        "theta": np.zeros(N),
        "Dtheta": np.zeros(N),
        "territory": np.sqrt(expon.rvs(scale=territory_scale, size=N) / np.pi)
    }

    history = []


    t = 0.0
    output_dir.mkdir(exist_ok=True)
    #h5_path = output_dir / f"{run}_simulation.h5"
    #h5file = h5py.File(h5_path, "w")
    #vlen_float = h5py.vlen_dtype(np.float32)
    #vlen_int = h5py.vlen_dtype(np.int32)

    #num_snapshots = (STEPS + snapshot_interval_steps - 1) // snapshot_interval_steps

    # reduced dataset for snapshots
    #dset_B = h5file.create_dataset("B", shape=(num_snapshots, L_h, L_w), dtype="int16",
    #                               compression="gzip", chunks=(1, L_h, L_w))
    #dset_time = h5file.create_dataset("time", shape=(num_snapshots,), dtype="float32")
    #dset_pos = h5file.create_dataset("lynx_pos", shape=(num_snapshots,), dtype=vlen_float)
    #dset_state = h5file.create_dataset("lynx_state", shape=(num_snapshots,), dtype=vlen_int)

    #snapshot_idx = 0

    for step in range(STEPS):

        reactions = [
            lambda: do_hare_death(B),
            lambda: do_hare_birth(B),
            lambda: do_hare_diffusion(B),
            lambda: do_predation(B, pos, mu),
            lambda: do_lynx_death(pos, state, params, compute_density(B)),
            lambda: do_lynx_birth(pos, state, params, compute_density(B)),
            lambda: lynx_movement_reaction(pos, state, params, compute_density(B), lambda_12)
        ]

        # Shuffle the reactions
        np.random.shuffle(reactions)

        # Apply them
        for reaction in reactions:
            result = reaction()
            if isinstance(result, tuple):
                pos, state, params = result
            else:
                B = result
        
        # extinction check
        if B.sum() == 0:
            print(f"Extinction: hares died out at step {step}, time {t:.2f}")
            break

        if len(pos) == 0:
            print(f"Extinction: lynx died out at step {step}, time {t:.2f}")
            break

        # update time
        t += DT
        n_state1 = np.sum(state == 1)
        n_state2 = np.sum(state == 2)

        history.append((t, B.sum(), n_state1, n_state2))
        # Append one timestep directly to disk
        #if step % snapshot_interval_steps == 0:
        #    dset_B[snapshot_idx] = B.astype(np.int16)
        #    dset_time[snapshot_idx] = t
        #    dset_pos[snapshot_idx] = pos.astype(np.float32).ravel()
        #    dset_state[snapshot_idx] = state.astype(np.int32)
        #    snapshot_idx += 1

    #h5file.close()
    return (np.array(history), B, pos, state)


def run_simulation(scenario_idx, run_idx, lambda_12_val):
    try:
        seed = int(time.time() * 1e6) % (2**32) + run_idx + scenario_idx * 1000
        np.random.seed(seed)

        print(f"Starting scenario {scenario_idx}, run {run_idx}, lambda_12={lambda_12_val}")

        output_dir = Path(f"/project/jnirody/Alissa/Results/fullSimulation_replicates/lambda12_{lambda_12_val}/")
        output_dir.mkdir(parents=True, exist_ok=True)

        history, B_final, lynx_final, state = simulate(output_dir, lambda_12_val, run_idx)

        job_id = os.environ.get("SLURM_JOB_ID", "local")

        np.savez_compressed(
            output_dir / f"scenario{scenario_idx}_run{run_idx}_job{job_id}.npz",
            history=np.array(history),
            B_final=B_final,
            lynx_pos=lynx_final,
            lynx_state=state
        )

        print(f"Finished scenario {scenario_idx}, run {run_idx}")

    except Exception as e:
        print(f"ERROR in scenario {scenario_idx}, run {run_idx}: {e}")
###################################### MAIN SIMULATION END ####################################################




if __name__ == "__main__":

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    # n_repeats per run
    n_reps = 100
    lambda_12_values = [0.0001, 0.00025, 0.0005, 0.001, 0.0015]

    scenario_idx = task_id // n_reps   # 0–4
    run_idx = task_id % n_reps         # 0–99

    lambda_12_val = lambda_12_values[scenario_idx]

    print(f"Task {task_id}: scenario {scenario_idx}, replicate {run_idx}")

    run_simulation(scenario_idx, run_idx, lambda_12_val)

    print("Simulation complete.")





