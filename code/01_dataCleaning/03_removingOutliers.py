import pandas as pd
import sys
from haversine import haversine, Unit
import numpy as np


### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = sys.argv[1]

def remove_high_speed_points(traj):
    """
    We have a few datapoints that are just unrealistic and clearly mistakes where the lynx will teleport 
    50km away within one 4-hour interval and then be right back where it was initially. We want to 
    remove those points from the trajectory. 
    
    Parameters:
        traj: pd.DataFrame that contains 'Lat', 'Long', and 'Time' columns

    Returns:
        traj: pd.DataFrame but where we've filtered out points with crazy speeds
    """
    max_speed_kmph=10 
    traj = traj.sort_values("Time").reset_index(drop=True)
    keep_mask = np.ones(len(traj), dtype=bool)

    coords = traj[['Lat', 'Long']].values
    times = traj['Time'].values

    for i in range(1, len(traj)):
        time_diff_hours = (times[i] - times[i - 1]) / np.timedelta64(1, 'h')
        if time_diff_hours <= 0:
            keep_mask[i] = False
            continue
        
        # using haversine to get the distances in kilometers
        dist_km = haversine(tuple(coords[i - 1]), tuple(coords[i]), unit=Unit.KILOMETERS)
        speed_kmph = dist_km / time_diff_hours

        if speed_kmph > max_speed_kmph:
            keep_mask[i] = False

    return traj.loc[keep_mask].reset_index(drop=True)

if __name__ == "__main__":
    data_path = f"{home_dir}/data/processed/filtered_lynx_processed.csv"
    output_path = f"{home_dir}/data/processed/final_lynx_df.csv"
    
    df = pd.read_csv(data_path, parse_dates=["Time"])
    df = df.groupby("ID", group_keys=False).apply(remove_high_speed_points)
    print("Removed ")

    # remove WSM025 because it's identical to WSM045 and we don't want duplicates
    df = df[df['ID'] != "WSM025"]
    print(f"Removed WSM025. Remaining unique IDs: {df['ID'].nunique()}")
    
    # fixing a whitespace issue for one of the lynx
    df['Sex'] = df['Sex'].astype(str).str.strip()  # remove leading/trailing spaces
    print("Unique sexes in data after stripping whitespace:", df['Sex'].unique())

    df.to_csv(output_path, index=False)






