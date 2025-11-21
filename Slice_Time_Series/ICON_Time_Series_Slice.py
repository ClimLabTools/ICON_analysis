from icon2vtk import icon_mesh

import os
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
from joblib import Parallel, delayed
#from copy import deepcopy

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
        Adapted copy of the functionality of the main function from icon_vtk_slice.py. The advantage of turning it into
        a method of the class is that mesh and time series are connected.

        parameters:
        plane: Either "surface" or a height value"

        '''

        df = pd.DataFrame(columns=['date', 'mean', 'var', 'min', 'max'])

        for filepath in tqdm(self.files):
            self.imesh.ds_icon = xr.open_dataset(filepath)
            #ncells = ds_icon.clon.shape[0]

            time_strs = self.imesh.ds_icon['time'].values.astype(str)
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
                elif variable == "w":
                    self.imesh.add_w(date_str=dstr)
                else:
                    print("add-function for variable name note implemented")
                    return None

                #print("---------------------------------------------------------")
                #print(f"Mesh cell_data: {self.imesh.mesh.cell_data}")
                #print("---------------------------------------------------------")
                #print(f"Mesh temp{self.imesh.topo['t_sk']}")
                #print("---------------------------------------------------------")

                if plane != "surface":

                    origin = (0, 0, plane)
                    normal = (0, 0, 1)
                    plane_params = {
                        "normal": normal,
                        "origin": origin,
                    }
                    self.slice = self.imesh.mesh.slice(normal=plane_params["normal"], origin=plane_params["origin"])
                else:
                    self.slice = self.imesh.topo

                #print(f"Slice cell_data{self.slice.cell_data}")
                #print("---------------------------------------------------------")
                #print(f"Slice temp: {self.slice['t_sk']}")
                #print("---------------------------------------------------------")


                if polygon != None:
                    points = self.slice.points[:, :2]
                    #print(f"Points: {len(points)}")
                    mask = np.array([polygon.contains(Point(p)) for p in points])

                    subset = self.slice.extract_points(mask)

                else:
                    subset = self.slice
                    #print(f"Subset 0: {subset}")



                #print(f"Subset 1: {subset}")
                #print(f"Slice cell_data: {subset.cell_data}")
                #print("---------------------------------------------------------")
                #print(f"Slice point_data: {subset.point_data}")
                #print("---------------------------------------------------------")
                #print(f"Slice temp: {subset['t_sk']}")
                #print("---------------------------------------------------------")

                if subset and variable in subset.cell_data:
                    values = subset.cell_data[variable]
                    values = np.where((values > value_limits[0]) & (values < value_limits[1]), values, np.nan)
                    values = values[~np.isnan(values)]

                    df.loc[len(df)] = [dstr, np.mean(values), np.var(values), np.min(values), np.max(values)]
                else:
                    print("subset for variable name note implemented")

                #print(subset)
                #print("---------------------------------------------------------")
                #print(subset.cell_data)
                #print("---------------------------------------------------------")


                #subset.plot(scalars=variable)
            #df.to_csv(f"./dfs/df_3_step{os.path.splitext(os.path.basename(filepath))[0]}.csv")

            #print(df)
            #print("---------------------------------------------------------")


        if save_as != None:
            df.to_csv(save_as)

        return df

def icon_ts_parallel(fgrid, fext, nlayers, filepaths, variable, polygon, plane, value_limits):

    TS = ICON_TS(fgrid=fgrid, fext=fext, nlayers=nlayers, filepaths=filepaths)

    df = TS.generate_time_series(variable=variable, polygon=polygon, plane=plane, value_limits=value_limits,
                                   save_as=None)
    return df

def main():
    # set path to dir containing the ICON output files
    filepaths = glob.glob(r"G:\NCs\*.nc") #(r"C:\Users\MS\OneDrive\Arbeit_HU\Tasks\Data\NCs\*.nc")[:2]
    print(filepaths)


    # Adapt paths to the location of the files with external parameters and grid
    fext = r"../../Data/external_parameter_icon_hef_DOM01_tiles.nc"
    fgrid = r"../../Data/hef_51m_DOM01.nc"

    # Set path to the polygon used to limit the grid (glacier shp or else)
    gdf_area = gpd.read_file(r"C:\Users\MS\OneDrive\Arbeit_HU\Tasks\2025_T05_Slice_Statistics\Shapes\Area_Outline.shp")#("../../Data/test.shp")
    polygon_area = gdf_area.geometry.union_all()  # merge multiple features if needed

    gdf_surf = gpd.read_file(r"C:\Users\MS\OneDrive\Arbeit_HU\Tasks\2025_T05_Slice_Statistics\Shapes\Sub_Area.shp")  # ("../../Data/test.shp")
    polygon_surf = gdf_surf.geometry.union_all()  # merge multiple features if needed


    file_groups = [filepaths[::2], filepaths[1::2]]

    res = Parallel(n_jobs=2)(
        (delayed(icon_ts_parallel)(fgrid, fext, 150, fg, variable="w", polygon=polygon_area, plane=2500, value_limits=[-np.inf, np.inf]) for fg in file_groups)
    )

    print(res)
    result = pd.concat(res)
    print(result)
    result.to_csv("./dfs/df_test_3.csv")


    '''
    TS1 = ICON_TS(fgrid=fgrid, fext=fext, nlayers=150, filepaths=filepaths)

    #TS1.cmesh.plot()

    start = time.time()
    start_1 = time.time()
    # Test with surface temperature
    df1 = TS1.generate_time_series(variable="w", polygon=polygon_surf, plane="surface", value_limits=[-np.inf, np.inf], save_as="./dfs/df_test_1_old.csv")
    print(df1)
    end_01 = time.time()
    print(f"Time for 1st test: {end_01 - start_1}")

    start_2 = time.time()
    # Test with temperature at 2500 meters with polygon
    df2 = TS1.generate_time_series(variable="w", polygon=polygon_area, plane=2500, value_limits=[-np.inf, np.inf], save_as="./dfs/df_test_2_old.csv")
    print(df2)
    end_02 = time.time()
    print(f"Time for 2nd test: {end_02 - start_2}")
    end = time.time()
    print(f"Total Time: {end - start}")
    '''


if __name__ == "__main__":
    main()