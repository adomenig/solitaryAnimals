import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import sys

### Getting the home directory from the bash script ##
if len(sys.argv) < 2:
    print("Usage: python process_lynx.py /path/to/home_directory")
    sys.exit(1)

home_dir = sys.argv[1]

data_path = f"{home_dir}/data/processed/lynx_initial_clean.csv"
output_path = f"{home_dir}/data/processed/filtered_lynx_processed.csv"
gps_data = pd.read_csv(data_path)

# parse the time
gps_data['Time'] = pd.to_datetime(gps_data['Time'], errors='coerce', utc=False)
gps_data = gps_data.dropna(subset=['Time'])  # drop bad timestamps

# constants
four_hours = pd.Timedelta(hours=4)
five_min = pd.Timedelta(minutes=5) # we set a 5 minute buffer for timestamps so we still count 3:55 as 4:00

def round_to_nearest_4hr(timestamp):
    """Round timestamp to nearest 4-hour mark (00:00, 04:00, 08:00, etc.)"""
    # get the the day 
    day = timestamp.floor('D')
    # calculate the hours since midnight
    hours_since_midnight = (timestamp - day).total_seconds() / 3600
    # round to the nearest 4-hour interval
    rounded_hours = round(hours_since_midnight / 4) * 4
    # create the new timestamp
    return day + pd.Timedelta(hours=rounded_hours)

def process_lynx(lynx_id):
    """
    This applies the new timesteps to our lynx. We make sure that the rounded time is within 
    +- 5 minutes of the original time. If it is, we apply the new rounded timestamp. 
    """
    lynx_data = gps_data[gps_data['ID'] == lynx_id].sort_values('Time').copy()
    if lynx_data.empty:
        return []

    selected_rows = []
    
    for _, row in lynx_data.iterrows():
        original_time = row['Time']
        rounded_time = round_to_nearest_4hr(original_time)
        
        # check if the original time is within +-5 minutes of the rounded time
        if abs(original_time - rounded_time) <= five_min:
            # Keep this point, but use the exact 4-hour mark as timestamp
            new_row = row.copy()
            new_row['Time'] = rounded_time
            selected_rows.append(new_row)
    
    return selected_rows

def verify_4_hour_intervals(df):
    """This is just a verification that all timestamps are at exact 4-hour intervals"""
    df['Time'] = pd.to_datetime(df['Time'])
    for lynx_id, group in df.groupby('ID'):
        # check hours are multiples of 4
        bad_hours = group[~group['Time'].dt.hour.isin([0, 4, 8, 12, 16, 20])]
        if not bad_hours.empty:
            print(f"ID {lynx_id} has bad hours:")
            print(bad_hours['Time'])
            return False
        
        # check minutes and seconds are zero
        bad_time = group[(group['Time'].dt.minute != 0) | (group['Time'].dt.second != 0)]
        if not bad_time.empty:
            print(f"ID {lynx_id} has non-zero minutes/seconds:")
            print(bad_time['Time'])
            return False
   
    print("All timestamps are properly aligned to 4-hour intervals!")
    return True

if __name__ == "__main__":
    # get the unique lynx IDs
    unique_lynx_ids = gps_data['ID'].unique()
    
    # process the lynx in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_lynx, unique_lynx_ids))
    
    # combine the results
    filtered_df = pd.concat([pd.DataFrame(rows) for rows in results if rows], ignore_index=True)

    # sort and verify
    filtered_df = filtered_df.sort_values(['ID', 'Time']).reset_index(drop=True)
    verify_4_hour_intervals(filtered_df)

    # save to CSV
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved filtered data to {output_path}")
