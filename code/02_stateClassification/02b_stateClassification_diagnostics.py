import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter, defaultdict

### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = Path(sys.argv[1])
sys.path.insert(0, str(home_dir))
import helper_functions  # type:ignore

# define paths
data_path = Path(f"{home_dir}/data/processed/stateClassification/final_lynx_with_states.csv")
out_path = Path(f"{home_dir}/outputs/classification_diagnostics/02_diagnostics")
out_path.mkdir(parents=True, exist_ok=True)

colorscheme = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255", "#c084d4"]

########## MEAN SQUARED DISPLACEMENT PLOT START #############################
def extract_state_segments(states, coords, times):
    """
    Extract continuous segments for each state, but we need to preserve coordinates and timestamps.
    Returns:
        segments[state] = list of dicts with keys:
            - xy
            - time
    """
    segments = {1: [], 2: []}

    in_state = None
    start = None

    for i in range(len(states)):

        if states[i] != in_state:

            # close previous segment
            if in_state is not None:
                segments[in_state].append({
                    "xy": coords[start:i],
                    "time": times[start:i]
                })

            in_state = states[i]
            start = i

    # close final segment
    if in_state is not None:
        segments[in_state].append({
            "xy": coords[start:],
            "time": times[start:]
        })

    return segments

def plot_statewise_msds(df, out_path):
    """
    Plot the MSDs for each state. We project to cartesian space first before calculating
    distances. Also, we account for time differences in time steps. 
    """
    state_info = [(1, 'Stationary'), (2, 'Exploratory')]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

    df = df.copy()
    df["Time"] = pd.to_datetime(df["Time"])

    for lynx_id, traj in df.groupby("ID"):

        traj = traj.sort_values("Time")

        coords = traj[['Lat', 'Long']].values
        states = traj['State'].values
        times = traj['Time'].values

        # project to meters
        xy_meters = helper_functions.project_to_alaska_albers(coords)

        # split into state segments including timestamps
        segments = extract_state_segments(states, xy_meters, times)

        for ax, (state_val, title) in zip(axs, state_info):

            for seg in segments[state_val]:

                if len(seg["xy"]) < 24:
                    continue

                xy = seg["xy"]
                t = seg["time"]

                max_lag = len(xy) // 2
                msd = helper_functions.compute_msd(xy, max_lag)
                msd = msd / 1e6  # m² → km²

                # compute REAL time lags
                t0 = t[0]
                lag_hours = np.array([
                    (t[i] - t0) / np.timedelta64(1, 'h')
                    for i in range(1, len(msd) + 1)
                ])

                if len(msd) > 20:
                    msd = msd[:-20]
                    lag_hours = lag_hours[:-20]

                ax.plot(lag_hours, msd,
                        color=colorscheme[state_val * 2],
                        alpha=0.3)

            ax.set_title(title)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Lag (hours)")
            ax.grid(True)

    axs[0].set_ylabel("MSD (km²)")

    plt.tight_layout()
    plt.savefig(out_path / "msd_by_state.png")
    print("Plotted statewise MSD.\n")
########## MEAN SQUARED DISPLACEMENT PLOT END #############################


################ VELOCITY/TURNING ANGLE PLOT START #############################
def turning_angles_planar(coords):
    """
    Turning angles in projected Cartesian space.
    """
    diffs = coords[1:] - coords[:-1]

    headings = np.arctan2(diffs[:, 1], diffs[:, 0])

    dtheta = headings[1:] - headings[:-1]
    dtheta = (dtheta + np.pi) % (2*np.pi) - np.pi

    return np.abs(dtheta)

def compute_velocity_and_turns(coords, times):
    # get the times
    times = pd.to_datetime(times)

    # project to cartesian space
    xy = helper_functions.project_to_alaska_albers(coords)

    # velocity
    dxy = xy[1:] - xy[:-1]
    distances = np.linalg.norm(dxy, axis=1) / 1000  # km

    dt = np.array([(times[i+1] - times[i]).total_seconds() / 3600 for i in range(len(times)-1)])
    velocities = distances / dt  # km/h

    # turning angles 
    turning_angles = turning_angles_planar(xy)

    return velocities, turning_angles

def extract_velocity_turn_by_state(df):
    out = { 'state1': {'velocities': [], 'turning_angles': []},
            'state2': {'velocities': [], 'turning_angles': []}}

    for lynx_id, traj in df.groupby("ID"):
        coords = traj[['Lat','Long']].values
        times = pd.to_datetime(traj['Time']).to_list()
        states = traj['State'].values

        velocities, turning_angles = compute_velocity_and_turns(coords, times)

        for state_val, key in [(1,'state1'),(2,'state2')]:
            idx = np.where(states[1:-1] == state_val)[0] 
            out[key]['velocities'].extend(velocities[idx]) 
            out[key]['turning_angles'].extend(turning_angles[idx])

    for key in out:
        out[key]['velocities'] = np.array(out[key]['velocities'])
        out[key]['turning_angles'] = np.array(out[key]['turning_angles'])

    return out

def plot_velocity_turn_heatmaps(vel_turn_data, out_path, n_angle_bins=36, n_vel_bins=40, max_vel=8):
    state_info = [('state1','Stationary'), ('state2','Exploratory')]
    fig, axs = plt.subplots(1,2, figsize=(12,5), sharey=True)

    # create custom colormaps that match the colors we used for our states earlier
    cmap_state1 = LinearSegmentedColormap.from_list("state1_cmap", ["white", colorscheme[2]])
    cmap_state2 = LinearSegmentedColormap.from_list("state2_cmap", ["white", colorscheme[4]])

    cmaps = [cmap_state1, cmap_state2]

    for ax, (key, title), cmap in zip(axs, state_info, cmaps):

        angles = vel_turn_data[key]['turning_angles']
        velocities = vel_turn_data[key]['velocities']

        mask = velocities < max_vel
        angles = angles[mask]
        velocities = velocities[mask]

        H, _, _ = np.histogram2d(angles, velocities, bins=[n_angle_bins, n_vel_bins], range=[[0,np.pi],[0,max_vel]], density=True)

        H[H <= 0] = 1e-10

        im = ax.imshow(
            H.T,
            origin='lower',
            aspect='auto',
            extent=[0,np.pi,0,max_vel],
            cmap=cmap,
            norm=mcolors.LogNorm()
        )
        ax.set_title(title)
        ax.set_xlabel("Turning angle (rad)")
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Density (log scale)")
    axs[0].set_ylabel("Velocity (km/h)")
    plt.tight_layout()
    plt.savefig(out_path / "velocity_turning_angle_dist.png", dpi=300)
    print(f"Plotted velocity and turning angle heatmaps for each state.\n")
################ VELOCITY/TURNING ANGLE PLOT END #############################



################ EXAMPLE TRAJECTORY PLOTS COLORED BY STATE START ##############
def plot_selected_trajectories(df, id_list, out_path):
    """
    Plots full trajectories for selected IDs, colored by state (1 and 2).
    Each figure corresponds to a single ID.
    """
    state_info = {2: "Exploratory", 1: "Stationary",}
    for lynx_id in id_list:
        traj_df = df[df["ID"] == lynx_id].copy()

        if len(traj_df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))

        # plot trajectory colored by state
        for state_val, title in state_info.items():
            state_traj = traj_df[traj_df["State"] == state_val][["Lat", "Long"]].values
            if len(state_traj) == 0:
                continue

            color = colorscheme[state_val*2]
            ax.scatter(state_traj[:, 1], state_traj[:, 0], color=color, s=20, alpha=0.7)

        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(f"Trajectory for ID {lynx_id}")
        plt.tight_layout()
        plt.savefig(out_path / f"trajectory_{lynx_id}.png")
        plt.close(fig)
        print(f"Plotted {lynx_id} trajectory.\n")
################ EXAMPLE TRAJECTORY PLOTS COLORED BY STATE END ##############

################ COMPUTING TRANSITION PROBABILITIES START ##################
def compute_transition_rates(df, dt_hours=4, id_col="ID", time_col="Time", state_col="State"):
    """
    Compute Markov transition probabilities per 4-hour timestep from our dataframe.
    """

    transitions_count = defaultdict(int)
    state_time_totals = Counter()
    transitions_detail = []

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    all_ids = df[id_col].unique()

    for lynx_id in all_ids:

        traj = df[df[id_col] == lynx_id].copy()
        traj = traj[[time_col, state_col]].sort_values(time_col).reset_index(drop=True)

        if len(traj) == 0:
            continue

        # regular grid
        t_start, t_end = traj[time_col].iloc[0], traj[time_col].iloc[-1]
        regular_times = pd.date_range(start=t_start, end=t_end, freq=f"{dt_hours}h")

        # backward interpolation if we don't have a data point
        df_interp = pd.merge_asof(
            pd.DataFrame({time_col: regular_times}),
            traj,
            on=time_col,
            direction="backward"
        )

        interp_states = df_interp[state_col].values

        # count transitions
        for i in range(len(interp_states) - 1):

            s0 = interp_states[i]
            s1 = interp_states[i + 1]

            if pd.isna(s0) or pd.isna(s1):
                continue

            state_time_totals[s0] += dt_hours
            transitions_count[(s0, s1)] += 1

            if s0 != s1:
                transitions_detail.append({
                    "Location": lynx_id,
                    "From": s0,
                    "To": s1,
                    "Time": df_interp[time_col].iloc[i + 1]
                })

        # final state time
        if len(interp_states) > 0 and not pd.isna(interp_states[-1]):
            state_time_totals[interp_states[-1]] += dt_hours

    # compute probabilities
    transition_probs = {}

    for (from_state, to_state), count in transitions_count.items():

        total_time_in_state = state_time_totals[from_state]
        num_intervals_in_state = total_time_in_state / dt_hours

        transition_probs[(from_state, to_state)] = count / num_intervals_in_state

    print(f"\nTransition Probabilities per {dt_hours}-hour Step")
    for (i, j), p in sorted(transition_probs.items()):
        print(f"P({i} → {j}) = {p:.4f}")

    
    return transition_probs, state_time_totals, transitions_detail
################# COMPUTING TRANSITION PROBABILITIES END ##################


if __name__ == "__main__":
    df = pd.read_csv(data_path, parse_dates=["Time"])

    # plotting median squared displacement split by state
    plot_statewise_msds(df, out_path)

    # velocity and turning angle distributions
    vel_turn_data = extract_velocity_turn_by_state(df)
    plot_velocity_turn_heatmaps(vel_turn_data, out_path)

    # plotting trajectories colored by state for selected lynx. Feel free to change
    selected_lynx = ["KOY025", "WSM034", "TET071"]
    plot_selected_trajectories(df, selected_lynx, out_path)

    # computing the transition probabilites from state 1 to 2 for our lynx.
    compute_transition_rates(df)

