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
        self.slice = None
        end = time.time()
        print(f"Mesh Object Generated In {end - start} Seconds...")

    def generate_time_series(self, variable, polygon=None, plane="surface", value_limits=[-np.inf, np.inf],
                             save_as=None):
        '''
        Adapted copy of the functionality of the main function from icon_vtk_slice.py.

        parameters:
        plane: Either "surface" or a height value"

        '''

        df = pd.DataFrame(columns=['date', 'mean', 'var', 'min', 'max'])

        for filepath in tqdm(self.files):
            ds_icon = xr.open_dataset(filepath)
            #ncells = ds_icon.clon.shape[0]

            time_strs = ds_icon['time'].values.astype(str)
            times = np.array([self.imesh._parse_time(time_str) for time_str in time_strs])

            for t in times:
                dstr = t.strftime('%Y%m%d %H:%M:%S')
                if variable == 't_sk':
                    self.imesh.add_tsk(date_str=dstr)
                elif variable == 'theta':
                    self.imesh.add_theta(date_str=dstr)
                elif variable == 't_2m':
                    self.imesh.add_t2m(date_str=dstr)
                elif variable == 'shfl_s':
                    self.imesh.add_shfl(date_str=dstr)
                elif variable == "temp":
                    self.imesh.add_temp(date_str=dstr)
                else:
                    print("add-function for variable name note implemented")
                    return None

                if plane != "surface":

                    origin = (0, 0, plane)
                    normal = (0, 0, 1)
                    plane_params = {
                        "normal": normal,
                        "origin": origin,
                    }

                    self.slice = self.imesh.mesh.slice(normal=plane_params["normal"], origin=plane_params["origin"])
                else:
                    self.slice = self.ptopo


                if polygon != None:
                    points = self.slice.points[:, :2]
                    mask = np.array([polygon.contains(Point(p)) for p in points])

                    subset = self.slice.extract_points(mask)

                else:
                    subset = self.slice

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
    df2 = TS1.generate_time_series(variable="temp", plane=2500, value_limits=[1, np.inf], save_as="../df_test_2.csv")
    print(df2)


if __name__ == "__main__":
    main()