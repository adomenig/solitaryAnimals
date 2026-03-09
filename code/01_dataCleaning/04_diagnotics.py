import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import seaborn as sns
from mpl_toolkits.basemap import Basemap


### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = Path(sys.argv[1])
sys.path.insert(0, str(home_dir))
import helper_functions # type:ignore

colorscheme = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255", "#c084d4"] 

############## BASIC DIAGNOSTICS PLOTTING STARAT ###################################
def plot_age_sex_distribution(df, out_path):
    """Plotting the distribution of Male vs Female and Juvenile vs Adult."""
    unique_individuals = df.drop_duplicates(subset="ID").copy()
    unique_individuals['Sex'] = unique_individuals['Sex'].str.strip()
    unique_individuals['Age'] = unique_individuals['Age'].str.strip()

    # count the numebr of males vs females and juveniles vs adults
    count_sex_df = unique_individuals['Sex'].value_counts().rename_axis('Sex').reset_index(name='Count')
    count_age_df = unique_individuals['Age'].value_counts().rename_axis('Age').reset_index(name='Count')
    count_age_df = count_age_df.replace({"Juvenille": "Juvenile"})

    fig, axes = plt.subplots(1, 2, figsize=(6, 5), sharey=True)

    # sex barplot
    colors_sex = [colorscheme[0] if s == 'M' else colorscheme[2] for s in count_sex_df['Sex']]
    axes[0].bar(count_sex_df['Sex'], count_sex_df['Count'], color=colors_sex, edgecolor=None)
    axes[0].set_xlabel("Sex")
    axes[0].set_ylabel("Number of Individuals")

    # age barplot
    colors_age = [colorscheme[4] if a == 'Adult' else colorscheme[6] for a in count_age_df['Age']]
    axes[1].bar(count_age_df['Age'], count_age_df['Count'], color=colors_age, edgecolor=None)
    axes[1].set_xlabel("Age")

    plt.tight_layout()
    plt.savefig(out_path / "Age_Sex_dist.png", dpi=300)
    print("Plotted the age and sex distribution of the lynx. \n")

def plot_daily_lynx_counts(df, out_path):
    df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Time'].dt.date

    # count the number of unique lynx per day
    daily_counts = df.groupby('Date')['ID'].nunique().reset_index()
    daily_counts.columns = ['Date', 'Unique_Lynx_Count']
    daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])

    plt.figure(figsize=(6, 3))
    ax = sns.scatterplot(
        x='Date',
        y='Unique_Lynx_Count',
        data=daily_counts,
        color=colorscheme[5],
        s=5,         
        alpha=0.5,  
        edgecolor=None
    )

    plt.xlabel("Date")
    plt.ylabel("Number of Lynx Tracked")
    plt.tight_layout()
    plt.savefig(out_path / "daily_lynx_counts_points.png", dpi=300)
    print("Plotted the daily counts of lynx with available data. \n")

############## BASIC DIAGNOSTICS PLOTTING END ###################################






############## LYNX TRAJECTORIES ON BASEMAP PLOTTING START ###################################
def plot_lynx_trajectories_on_basemap(df, out_path=None):
    """Plotting all the lynx trajectories over a basemap of Alaska"""
    # compute buffered bounding box for map extent
    buffer_lat = 2.0  # degrees
    buffer_long = 4.0
    min_lon, max_lon = df["Long"].min(), df["Long"].max()
    min_lat, max_lat = df["Lat"].min(), df["Lat"].max()

    # create the map
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_frame_on(False)

    m = Basemap(
        projection='merc',
        llcrnrlat=min_lat - buffer_lat,
        urcrnrlat=max_lat + buffer_lat,
        llcrnrlon=min_lon - buffer_long,
        urcrnrlon=max_lon + buffer_long,
        resolution='i',
        ax=ax
    )

    m.drawcoastlines(color="#d3d3d3")
    m.drawcountries(color="#d3d3d3")
    m.drawmapboundary(fill_color="white")
    m.fillcontinents(color="#d3d3d3", lake_color="white")

    # plot each trajectory
    for _, group in df.groupby("ID"):
        group = group.sort_values("Time")
        x, y = m(group["Long"].values, group["Lat"].values)
        m.plot(
            x, y,
            linewidth=0.6,
            color="#000000",
        )

    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / "lynx_over_basemap.png", dpi=300, bbox_inches="tight")
    print("Plotted lynx trajectories on basemap. \n")
 ############## LYNX TRAJECTORIES ON BASEMAP PLOTTING END ###################################





############## MSD PLOTTING START ###################################
def calculate_for_lynx(args):
    """Helper function for plotting the lynx msds in parallel."""
    lynx_id, traj, max_lag_steps = args
    try:
        lags, msd = calculate_msd(traj, max_lag_steps=max_lag_steps)
        return lynx_id, lags, msd
    except Exception as e:
        print(f"Error with {lynx_id}: {e}")
        return lynx_id, np.array([]), np.array([])

def calculate_msd(traj, max_lag_steps=None):
    latlon = np.radians(traj[['Lat', 'Long']].values.astype(np.float64))
    times = traj['Time'].values.astype('datetime64[s]').astype(np.float64) / 3600  # convert to hours

    lags_hours = []
    msds = []

    for lag in range(1, max_lag_steps + 1):
        # vectorized: points separated by lag
        lat1 = latlon[:-lag, 0]
        lon1 = latlon[:-lag, 1]
        lat2 = latlon[lag:, 0]
        lon2 = latlon[lag:, 1]

        dt = times[lag:] - times[:-lag]
        valid_mask = dt > 0  # filter out non-positive time differences

        if np.sum(valid_mask) == 0:
            continue

        # calculate the distances for all pairs at once -> we need to convert it to haversine 
        # distances since we're working with latitude and longitudes
        dists = helper_functions.haversine_vectorized(lat1[valid_mask], lon1[valid_mask], lat2[valid_mask], lon2[valid_mask])
        squared_displacements = dists**2
        lags_hours.append(np.mean(dt[valid_mask]))
        msds.append(np.mean(squared_displacements))
    return np.array(lags_hours), np.array(msds)



def plot_all_lynx_msds(df, out_path, max_workers=10):
    """Plot the MSDs of eevery lynx on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # estimate a safe max lag
    max_len = df.groupby("ID").size().max()
    max_lag_steps = max(max_len - 100, 1) # set an early cutoff to the MSDs so that we're averaging over at least 100 points
    
    all_lags_dict = {}  # store raw MSDs 
    n_plotted = 0
    # plot individual MSDs in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(calculate_for_lynx, (lynx_id, group.copy(), max_lag_steps))
            for lynx_id, group in df.groupby("ID")
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel MSDs"):
            lynx_id, lags, msd = future.result()
            if len(lags) > 0:
                n_plotted += 1
                all_lags_dict[lynx_id] = (lags, msd)
                ax.plot(lags, msd, color="#CECBCB", linewidth=0.9)
    ax.set_xlabel("Lag Time (hours)", fontsize=16, labelpad=10)
    ax.set_ylabel("Mean Squared Displacement (km²)", fontsize=16, labelpad=10)
    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    fig.tight_layout()
    plt.savefig(out_path / "MSDs.png", dpi=300)
    print(f"Plotted MSDS for {n_plotted} Lynx\n")
############## MSD PLOTTING END ###################################




############## SELETED LYNX TRAJECTORY PLOTTING START ###################################
def plot_selected_lynx_trajectories(df, selected_lynx, out_path, buffer_frac=0.05):
    """
    Plots each selected lynx trajectory as a subplot in a single figure.
    """
    colors = [colorscheme[1], colorscheme[8], colorscheme[3]]

    # Create the subplots
    fig, axes = plt.subplots(
        1, len(selected_lynx),
        figsize=(6 * len(selected_lynx), 6),  # width scales with number of lynx
    )

    if len(selected_lynx) == 1:
        axes = [axes]

    # Plot each lynx individually
    for i, (ax, lynx_id) in enumerate(zip(axes, selected_lynx)):
        lynx_data = df[df["ID"] == lynx_id]

        ax.scatter(
            lynx_data["Long"],
            lynx_data["Lat"],
            s=10,
            alpha=0.7,
            color=colors[i % len(colors)],
        )

        # Compute individual bounds with buffer
        min_lon, max_lon = lynx_data["Long"].min(), lynx_data["Long"].max()
        min_lat, max_lat = lynx_data["Lat"].min(), lynx_data["Lat"].max()

        lon_buffer = (max_lon - min_lon) * buffer_frac
        lat_buffer = (max_lat - min_lat) * buffer_frac
        ax.set_xlim(min_lon - lon_buffer, max_lon + lon_buffer)
        ax.set_ylim(min_lat - lat_buffer, max_lat + lat_buffer)

        # Keep natural aspect ratio based on axes
        ax.set_aspect((max_lon - min_lon) / (max_lat - min_lat))

        ax.grid(False)
        ax.tick_params(axis="both", which="major", labelsize=20, length=6, width=1.5, direction="out")

        ax.set_title(f"Lynx {lynx_id}", fontsize=16)

    axes[0].set_ylabel("Latitude", fontsize=18)
    for ax in axes:
        ax.set_xlabel("Longitude", fontsize=18)

    plt.tight_layout()
    out_file = Path(out_path) / "selected_lynx_trajectories_subplots.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Plotted selected lynx trajectories for lynx: {', '.join(selected_lynx)} \n")
############## SELETED LYNX TRAJECTORY PLOTTING END ###################################




############## PLOTTING INSTANTANEOUS VELOCITY START ###################################
def compute_velocity(df):
    """
    Compute velocity between two consecutive points for each lynx.
    Assumes that the df has ['ID', 'Time', 'Long', 'Lat'] columns.
    Returns a new DataFrame with an additional 'Velocity_kmh' column.
    """
    df = df.sort_values(["ID", "Time"])
    all_groups = []

    for lynx_id, group in df.groupby("ID"):
        group = group.copy()  
        lons = group["Long"].values
        lats = group["Lat"].values
        times = group["Time"].values.astype("datetime64[s]").astype(float)  # seconds
        velocities = [0.0]  # first point = 0

        for i in range(1, len(group)):
            dt = times[i] - times[i-1]
            if dt == 0:
                vel = 0
            else:
                dist = helper_functions.haversine_vectorized(lats[i-1], lons[i-1], lats[i], lons[i])
                vel = dist / (dt / 3600.0)  # km/h
            velocities.append(vel)

        group["Velocity_kmh"] = velocities
        all_groups.append(group)

    df_vel = pd.concat(all_groups)
    df_vel = df_vel.sort_values(["ID", "Time"])  # restore the order
    return df_vel

def plot_lynx_velocity(df, selected_lynx, out_path):
    """
    Plotting instantaneous velocities over time.
    """
    n = len(selected_lynx)

    fig, axes = plt.subplots(
        1, n,
        figsize=(10, 3),   # make it wide enough
        sharey=True
    )

    if n == 1:
        axes = [axes]

    colors = [colorscheme[3], colorscheme[1], colorscheme[8]]
    # global y-limits
    sub = df[df["ID"].isin(selected_lynx)]
    ymax = sub["Velocity_kmh"].max()
    for i, (ax, lynx_id) in enumerate(zip(axes, selected_lynx)):
        lynx_data = df[df["ID"] == lynx_id]

        ax.scatter(
            lynx_data["Time"],
            lynx_data["Velocity_kmh"],
            s=2,
            color=colors[i % len(colors)]
        )

        # only leftmost plot gets y-axis label
        if i == 0:
            ax.set_ylabel("Velocity (km/h)", fontsize=18)
        else:
            ax.tick_params(labelleft=False)  # hide y tick labels

        ax.set_xlabel("Time", fontsize=18)

        xmin = lynx_data["Time"].min()
        xmax = lynx_data["Time"].max()
        ax.set_xlim(xmin, xmax)

        # set ticks exactly at first and last data points
        ax.set_xticks([xmin, xmax])
        ax.set_ylim(0, ymax)
        # align tick labels so they stay inside the plot 
        labels = ax.get_xticklabels()

        if len(labels) == 2:
            labels[0].set_horizontalalignment("left")   # make the left label extend inward
            labels[1].set_horizontalalignment("right")  # make the right label extend inward
        ax.tick_params(axis="both", labelsize=16)
    plt.tight_layout()
    out_file = Path(out_path) / "selected_lynx_velocity.png"
    plt.savefig(out_file, dpi=300)
    plt.close()

    print(f"Plotted instantaneous velocities over time for selected lynx: {', '.join(selected_lynx)} \n")
############## PLOTTING INSTANTANEOUS VELOCITY END ###################################




if __name__ == "__main__":
    data_path = Path(f"{home_dir}/data/processed/final_lynx_df.csv")
    out_path = Path(f"{home_dir}/outputs/data_diagnostics")
    df = pd.read_csv(data_path, parse_dates=["Time"])

    # plotting the distribution of age and sex in the lynx
    plot_age_sex_distribution(df, out_path)

    # plot a daily count of how many lynx we have data for
    plot_daily_lynx_counts(df, out_path)

    # plotting all the lynx trajectories on a basemap of Alaska + Western Canada
    plot_lynx_trajectories_on_basemap(df, out_path)

    # plotting the msds of all the lynx trajectories
    plot_all_lynx_msds(df, out_path)

    # selecting three representative lynx. These IDs can be replaced with 
    # any other IDs of interest. 
    selected_lynx = ["TET064", "TET071", "TET042"]
     
    # plotting their trajectories individually
    plot_selected_lynx_trajectories(df, selected_lynx, out_path)
    
    # plotting the velocity over time distributions of these lynx
    velocity_df = compute_velocity(df)
    plot_lynx_velocity(velocity_df, selected_lynx, out_path)

    print(f"Finished plotting diagnostics. All the plots can be found in {out_path}. \n")




    

