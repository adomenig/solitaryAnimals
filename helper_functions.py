### This file contains some of the helper functions we re-use accross code files. ###
import numpy as np
from pyproj import Transformer

albers = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0  # km

    # convert to radians 
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c  # km


def compute_msd(xy, max_lag_steps):
    """
    Standard median squared displacement.
    """
    msd = []
    for lag in range(1, max_lag_steps + 1):
        disp = xy[lag:] - xy[:-lag]
        squared_disp = np.sum(disp**2, axis=1)
        msd.append(np.median(squared_disp))
    return np.array(msd)


def project_to_alaska_albers(coords):
    """
    coords: (N,2) array in [lat, lon]
    returns: (N,2) array in meters (x,y)
    """
    lon = coords[:, 1]
    lat = coords[:, 0]
    x, y = albers.transform(lon, lat)
    return np.column_stack([x, y])