from os import times_result

import xarray as xr
import numpy as np
import metpy
from metpy.units import units
import pyvista as pv
from datetime import datetime, timedelta
import geopandas as gpd
import pandas as pd
import glob
from shapely.geometry import Polygon, Point

def add_theta(mesh, ncells, nlayers, date_str=None):
    if date_str is None:
        print("add_theta: Please provide a date")
        pass
    if isinstance(date_str, str):
        temp = np.ndarray([ncells * nlayers]).astype(float)
        pres = np.ndarray([ncells * nlayers]).astype(float)
        idx = _get_time(date_str)
        _temp = ds_icon['temp'][idx, :, :].values
        _pres = ds_icon['pres'][idx, :, :].values
        for i in range(0, ncells, 1):
            for z in range(0, nlayers):
                temp[(i * nlayers) + z] = _temp[(nlayers) - z, i]
                pres[(i * nlayers) + z] = _pres[(nlayers) - z, i]

        mesh.cell_data['theta'] = metpy.calc.potential_temperature(pres * units.Pa,
                                                                        temp * units.kelvin).magnitude
        return mesh

def _parse_time(time_str):
    date_part = time_str.split('.')[0]
    fractional_part = time_str.split('.')[1]
    fractional_days = float('0.' + fractional_part)

    # Parse the date part
    dt = datetime.strptime(date_part, '%Y%m%d')

    # Add fractional day as a timedelta
    dt += timedelta(days=fractional_days)

    # Round the datetime to the closest minute
    rounded_dt = dt.replace(second=0, microsecond=0) + timedelta(minutes=1) if dt.second >= 30 else dt.replace(
        second=0, microsecond=0)

    return rounded_dt


def _get_time(target_date_str):
    # Extract the time variable as an array of strings (e.g., ['20230906.123456', ...])
    time_strs = ds_icon['time'].values.astype(str)
    # Convert all time strings to datetime objects
    times = np.array([_parse_time(time_str) for time_str in time_strs])

    # Convert the target date to a datetime object
    target_date = datetime.strptime(target_date_str, '%Y%m%dT%H:%M:%S')

    # Find the index of the closest date by calculating the absolute difference
    time_diffs = np.abs(times - target_date)
    closest_index = np.argmin(time_diffs)
    return closest_index


# ---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
mesh = pv.read('/home/sauterto/icon2vtk/icon2vtk/icon_mesh.vtk')
filepaths = glob.glob("/data/projects/hefex/output/v3/exp_R3B15_51m/LES_DOM01_ML_0001.nc")

# --- 1. Load the shapefile (polygon)
gdf = gpd.read_file("test.shp")
polygon = gdf.geometry.union_all()  # merge multiple features if needed
# Select subset
#polygon = None

# Create empty DataFrame
df = pd.DataFrame(columns=['date', 'mean','var','min','max'])

# 2. Define the slicing plane (e.g., z-slice)
z_height = 2500.0
origin = (0, 0, z_height)
normal = (0, 0, 1)

print('Start')

for filepath in filepaths:
    print(filepath)
    ds_icon = xr.open_dataset(filepath)
    ncells = ds_icon.clon.shape[0]

    time_strs = ds_icon['time'].values.astype(str)
    times = np.array([_parse_time(time_str) for time_str in time_strs])

    for t in times:
        dstr = t.strftime('%Y%m%dT%H:%M:%S')
        mesh = add_theta(mesh, ncells, 50, date_str=dstr)

        slice_plane = mesh.slice(normal=normal, origin=origin)

        # --- 3. Filter slice points that lie inside the polygon
        if polygon != None:
            points = slice_plane.points[:, :2]  # drop Z (assumed flat)
            mask = np.array([polygon.contains(Point(p)) for p in points])

            # --- 4. Extract the subset
            subset = slice_plane.extract_points(mask)

            # Optional: visualize or save
            subset.save(f"slice_{dstr}.vtk")
        else:
            subset = slice_plane
            subset.save(f"slice_{dstr}.vtk")

        # === After closing: compute final statistics
        if subset and 'theta' in subset.cell_data:
            values = subset.cell_data['theta']
            values = values[~np.isnan(values)]  # Clean NaNs

            # Add values one by one
            df.loc[len(df)] = [dstr, np.mean(values), np.var(values), np.min(values), np.max(values)]
            print(df.iloc[-1])
