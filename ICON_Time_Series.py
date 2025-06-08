from icon2vtk import icon_mesh

import numpy as np
import pandas as pd
import glob
import geopandas as gpd
from tqdm import tqdm
import xarray as xr
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point
import pyvista as pv
import time
import metpy
from metpy import units
from pyproj import Transformer
from pyvista import CellType

############### add-functions from the imesh-class ########################################

def add_theta(mesh, ncells, nlayers, ds_icon, date_str=None):
    if date_str is None:
        print("add_theta: Please provide a date")
        pass
    if isinstance(date_str, str):
        temp = np.ndarray([ncells * nlayers]).astype(float)
        pres = np.ndarray([ncells * nlayers]).astype(float)
        idx = _get_time(date_str, ds_icon)
        _temp = ds_icon['temp'][idx, :, :].values
        _pres = ds_icon['pres'][idx, :, :].values
        for i in range(0, ncells, 1):
            for z in range(0, nlayers):
                temp[(i * nlayers) + z] = _temp[(nlayers) - z, i]
                pres[(i * nlayers) + z] = _pres[(nlayers) - z, i]

        mesh.cell_data['theta'] = metpy.calc.potential_temperature(pres * units.Pa,
                                                                        temp * units.kelvin).magnitude
        return mesh

def add_temp(mesh, ncells, nlayers, ds_icon, date_str=None):
    if date_str is None:
        print("add_temp: Please provide a date")
        pass
    if isinstance(date_str, str):
        temp = np.ndarray([ncells * nlayers]).astype(float)
        idx = _get_time(date_str, ds_icon)
        # Get u and v
        _temp = ds_icon['temp'][idx, :, :].values
        for i in range(0, ncells, 1):
            for z in range(0, nlayers):
                temp[(i * nlayers) + z] = _temp[(nlayers - 1) - z, i]
        mesh.cell_data['temp'] = temp

    return mesh

def add_tsk(mesh, ds_icon, date_str=None):
    if date_str is None:
        print("add_tsk: Please provide a date")
        pass
    if isinstance(date_str,str):
        idx = _get_time(date_str, ds_icon)
        mesh.cell_data['t_sk'] = ds_icon['t_sk'][idx, :].values

        return mesh

def add_t2m(mesh, ds_icon, date_str=None):
    if date_str is None:
        print("add_tsk: Please provide a date")
        pass
    if isinstance(date_str,str):
        idx = _get_time(date_str, ds_icon)
        mesh.cell_data['t_2m'] = ds_icon['t_2m'][idx, 0, :].values

        return mesh

def add_shfl(mesh, ds_icon, date_str=None):
    if date_str is None:
        print("add_tsk: Please provide a date")
        pass
    if isinstance(date_str,str):
        idx = _get_time(date_str, ds_icon)
        print(idx, ds_icon["shfl_s"].shape)
        mesh.cell_data['shfl_s'] = ds_icon['shfl_s'][idx, :].values

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


def _get_time(target_date_str, ds_icon):
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

###########################################################################################

class ICON_TS:

    def __init__(self, fgrid, fext, nlayers, filepaths):
        '''
        Class that stores the surface and 3D grid for time-series generation
        '''
        start = time.time()
        self.files = filepaths
        self.imesh = icon_mesh(fgrid=fgrid, ficon=filepaths[0], fext=fext, nlayers=nlayers)
        self.ptopo = self.imesh.create_topo()
        self.cmesh = self.imesh.create_grid()
        self.timeseries = None
        self.plane = None
        end = time.time()
        print(f"Mesh Object Generated In {end - start} Seconds...")

    def generate_time_series(self, variable, polygon=None, plane="surface", value_limits=[-np.inf, np.inf],
                             save_as=None):
        '''
        Adapted copy of the functionality of the main function from icon_vtk_slice.py.

        parameters:
        plane: Either "surface" or a height value"

        '''

        if plane != "surface":

            origin = (0, 0, plane)
            normal = (0, 0, 1)
            plane_params = {
                "normal": normal,
                "origin": origin,
            }

            self.plane = self.imesh.mesh.slice(normal=normal, origin=origin)
        else:
            self.plane = self.imesh.topo


        df = pd.DataFrame(columns=['date', 'mean', 'var', 'min', 'max'])

        for filepath in tqdm(self.files):
            ds_icon = xr.open_dataset(filepath)
            ncells = ds_icon.clon.shape[0]

            time_strs = ds_icon['time'].values.astype(str)
            times = np.array([_parse_time(time_str) for time_str in time_strs])

            for t in times:
                dstr = t.strftime('%Y%m%dT%H:%M:%S')
                if variable == 't_sk':
                    mesh = add_tsk(self.ptopo, ds_icon=ds_icon, date_str=dstr)
                    mesh = self.plane.sample(mesh)
                elif variable == 'theta':
                    mesh = add_theta(self.cmesh, ds_icon=ds_icon, ncells=ncells, nlayers=self.imesh.nlayers, date_str=dstr)
                    mesh = self.plane.sample(mesh)
                elif variable == 't_2m':
                    mesh = add_t2m(self.ptopo, ds_icon=ds_icon, date_str=dstr)
                    mesh = self.plane.sample(mesh)
                elif variable == 'shfl_s':
                    mesh = add_shfl(self.ptopo, ds_icon=ds_icon, date_str=dstr)
                    mesh = self.plane.sample(mesh)
                elif variable == "temp":
                    mesh = add_temp(self.cmesh, ds_icon=ds_icon, ncells=ncells, nlayers=self.imesh.nlayers, date_str=dstr)
                    mesh = self.plane.sample(mesh)
                else:
                    print("add-function for variable name not implemented")
                    return None


                if polygon != None:
                    points = mesh.points[:, :2]
                    mask = np.array([polygon.contains(Point(p)) for p in points])

                    subset = mesh.extract_points(mask)

                else:
                    subset = mesh

                if subset and variable in subset.cell_data:
                    values = subset.cell_data[variable]
                    values = np.where((values > value_limits[0]) & (values < value_limits[1]), values, np.nan)
                    values = values[~np.isnan(values)]

                    df.loc[len(df)] = [dstr, np.mean(values), np.var(values), np.min(values), np.max(values)]

        if save_as != None:
            df.to_csv(save_as)

        return df


def main():
    # set path to dir containing the ICON output files
    filepaths = glob.glob(r"G:\NCs\*.nc")
    print(filepaths)

    # Adapt paths to the location of the files with external parameters and grid
    fext = r"../../Data/external_parameter_icon_hef_DOM01_tiles.nc"
    fgrid = r"../../Data/hef_51m_DOM01.nc"

    # Set path to the polygon used to limit the grid (glacier shp or else)
    gdf = gpd.read_file("../../Data/test.shp")
    polygon = gdf.geometry.union_all()  # merge multiple features if needed

    TS1 = ICON_TS(fgrid=fgrid, fext=fext, nlayers=50, filepaths=filepaths)

    #TS1.cmesh.plot()

    # Test with surface temperature
    df1 = TS1.generate_time_series(variable="t_sk", polygon=polygon, plane="surface", value_limits=[1, np.inf], save_as="../df_test_1.csv")
    print(df1)

    # Test with temperature at 2500 meters with polygon
    df2 = TS1.generate_time_series(variable="temp", plane=2500, polygon=polygon, value_limits=[1, np.inf], save_as="../df_test_2.csv")
    print(df2)


if __name__ == "__main__":
    main()