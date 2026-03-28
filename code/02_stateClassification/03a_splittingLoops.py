import sys
from haversine import haversine, Unit
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.io as pio
from scipy.spatial import ConvexHull
pio.renderers.default = "browser"
import matplotlib.pyplot as plt
from pyproj import Transformer


### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = Path(sys.argv[1])
sys.path.insert(0, str(home_dir))
import helper_functions # type:ignore

# define paths
data_path = Path(f"{home_dir}/data/processed/stateClassification/final_lynx_with_states.csv")
out_path = Path(f"{home_dir}/data/processed/stateClassification")
out_path.mkdir(parents=True, exist_ok=True)

colorscheme = ["#8fd7d7", "#00b0be", "#ff8ca1", "#f45f74", "#bdd373", "#98c127", "#ffcd8e", "#ffb255", "#c084d4"] 


def compute_state1_convex_hulls(df: pd.DataFrame) -> np.ndarray:
    """
    Compute convex hull areas for all state 1 points for all lynx in Alaska,
    using the Alaska Albers projection (good for northern latitudes)
    """
    state1_areas = []
    lynx_ids = df["ID"].unique()

    # since these points are in alaska, we want to use the Alaska Albers projection (EPSG:3338 - NAD83 / Alaska Albers)
    transformer = Transformer.from_crs("epsg:4326", "epsg:3338", always_xy=True)  # lon, lat -> projected x, y

    for lynx_id in lynx_ids:
        traj = df[df["ID"] == lynx_id].reset_index(drop=True)
        state_mask = traj["State"] == 1
        state1_points = traj[state_mask][["Lat", "Long"]].values

        # project lat/lon to x/y in meters
        lons = state1_points[:, 1]
        lats = state1_points[:, 0]
        x, y = transformer.transform(lons, lats)
        coords = np.column_stack([x, y])
        
        # calculate the convex hull
        hull = ConvexHull(coords)
        area_km2 = hull.volume / 1e6  # convert m² -> km²
        state1_areas.append(area_km2)
    
    state1_areas = np.array(state1_areas)
    state1_areas = state1_areas[state1_areas > 0]
    return state1_areas

def identify_and_split_loops(df, territory_radius_km):
    """
    Split State 2 excursions into outbound (2) and return (3) phases
    if the lynx returns near its starting location.
    """
    df = df.copy().sort_values(["ID", "Time"])
    updated_dfs = []

    for lynx_id, traj in df.groupby("ID"):
        traj = traj.reset_index(drop=True)
        states = traj["State"].values.copy()
        lats = traj["Lat"].values
        lons = traj["Long"].values

        # identify contiguous State 2 segments
        is_state2 = states == 2
        diff = np.diff(np.r_[0, is_state2.astype(int), 0])
        seg_starts = np.where(diff == 1)[0]
        seg_ends = np.where(diff == -1)[0] - 1


        for seg_start, seg_end in zip(seg_starts, seg_ends):
            start_coord = np.array([lats[seg_start], lons[seg_start]])
            end_coord = np.array([lats[seg_end], lons[seg_end]])

            # distance from start to end of state 2 segment
            start_end_dist = haversine(tuple(start_coord), tuple(end_coord), unit=Unit.KILOMETERS)
            # if the lynx is close enough to where it started when it went in the exploratory phase
            # to where it ends, then we assume it came back home so it's a loop. If not, then it's 
            # a territory swith.
            if start_end_dist > territory_radius_km: 
                continue  # not a loop

            # Vectorized distance to start and end
            seg_coords = np.column_stack((lats[seg_start:seg_end+1], lons[seg_start:seg_end+1]))
            dist_to_start = np.array([haversine(tuple(start_coord), tuple(c), unit=Unit.KILOMETERS) 
                                      for c in seg_coords])
            dist_to_end = np.array([haversine(tuple(end_coord), tuple(c), unit=Unit.KILOMETERS) 
                                    for c in seg_coords])

            combined_dist = dist_to_start + dist_to_end
            split_idx = seg_start + combined_dist.argmax()

            # assign the outbound and return states
            states[seg_start:split_idx+1] = 2
            states[split_idx+1:seg_end+1] = 3

        traj["State_Loop_Split"] = states
        updated_dfs.append(traj)

    return pd.concat(updated_dfs).sort_values(["ID", "Time"])

if __name__ == "__main__":
    df = pd.read_csv(data_path, parse_dates=["Time"])

    # get the convex hull of state 1 and take the 75th percentile to be 
    # a good assumption of how big most territories are. We use that to 
    # classify when a lynx actually comes back to the same territory vs when 
    # it's swithing to a new one after it goes on excursions
    state1_areas = compute_state1_convex_hulls(df)
    p95 = np.percentile(state1_areas, 95)
    radius = np.sqrt(p95 / np.pi)

    df = identify_and_split_loops(df, radius)
    df.to_csv(out_path / "final_lynx_with_states.csv", index=False)
