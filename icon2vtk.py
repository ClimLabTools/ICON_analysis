import time, os
import xarray as xr

import numpy as np
import metpy
from metpy.units import units

import pyvista as pv
from pyvista import CellType

from datetime import datetime, timedelta
from pyproj import Transformer

class icon_mesh():

    def __init__(self, fgrid, ficon, fext, nlayers):
        ds_grid = xr.open_dataset(fgrid)
        ds_icon = xr.open_dataset(ficon)
        ds_ext = xr.open_dataset(fext)

        self.clon_vertices = np.rad2deg(ds_grid.clon_vertices.values)
        self.clat_vertices = np.rad2deg(ds_grid.clat_vertices.values)
        self.ncells, nv = self.clon_vertices.shape[0], self.clon_vertices.shape[1]

        self.height_half = ds_icon.sizes['height_3']
        self.height_full = ds_icon.sizes['height']
        self.nlayers = nlayers
        self.nfaces = self.nlayers + 1

        # Grid variables
        self.np_points = np.ndarray([3 * (self.ncells * self.nfaces), nv])
        self.cell_type = np.ndarray(self.ncells * self.nlayers).astype(object)
        self.cells = np.ndarray([self.ncells * self.nlayers, 25]).astype(int)

        # height info z_ifc :: half-level, z_mc :: full-level
        self.z_ifc = np.ndarray([self.ncells * self.nfaces]).astype(float)
        self.z_mc = np.ndarray([self.ncells * self.nlayers]).astype(float)

        # at cell center
        self.w = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.u = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.v = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.temp = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.qv = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.qc = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.pres = np.ndarray([self.ncells * self.nlayers]).astype(float)

        # Auxiliary variable
        self.idx = 0

        self.ds_icon = ds_icon
        self.ds_ext = ds_ext
        self.z_ifc = ds_icon.z_ifc[(self.height_half-self.nfaces):self.height_half].values

    def _parse_time(self,time_str):
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

    def print_time(self):
        # Extract the time variable as an array of strings (e.g., ['20230906.123456', ...])
        time_strs = self.ds_icon['time'].values.astype(str)
        # Convert all time strings to datetime objects
        times = np.array([self._parse_time(time_str) for time_str in time_strs])
        #print(times)

    def _get_time(self, target_date_str):
        # Extract the time variable as an array of strings (e.g., ['20230906.123456', ...])
        time_strs = self.ds_icon['time'].values.astype(str)
        # Convert all time strings to datetime objects
        times = np.array([self._parse_time(time_str) for time_str in time_strs])

        # Convert the target date to a datetime object
        target_date = datetime.strptime(target_date_str, '%Y%m%d %H:%M:%S')

        # Find the index of the closest date by calculating the absolute difference
        time_diffs = np.abs(times - target_date)
        closest_index = np.argmin(time_diffs)
        return closest_index

    def _add_vertice(self, _lon, _lat, _z):
        self.np_points[self.idx, 0] = _lon
        self.np_points[self.idx, 1] = _lat
        self.np_points[self.idx, 2] = _z
        self.idx += 1

    def face_to_center(self, _var, date_str=None):
        if date_str is None:
            print("face_to_center: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            _var = self.ds[_var][idx, :, :].values
            _nvar = _var.copy()
            for i in range(0, self.ncells, 1):
                for z in range(1, self.nfaces):
                    _nvar[(i * self.nlayers) + (z - 1)] = (_var[self.nlayers - z - 1, i] + _var[self.nlayers - z, i]) / 2
        return _nvar

    def add_wind_vector(self, date_str=None):
        if date_str is None:
            print("add_wind_vector: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            _u = self.ds['u'][idx, :, :].values
            _v = self.ds['v'][idx, :, :].values
            # Interpolate w to cell center
            _w = self.face_to_center('w',date_str)
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.u[(i * self.nlayers) + z] = _u[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]
                    self.v[(i * self.nlayers) + z] = _v[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]
            self.mesh.cell_data['w'] = self.w
            self.mesh.cell_data['u'] = self.u
            self.mesh.cell_data['v'] = self.v
            self.mesh.cell_data['wind'] = np.column_stack((self.u, self.v, self.w))

    def add_theta(self, date_str=None):
        if date_str is None:
            print("add_theta: Please provide a date")
            pass
        if isinstance(date_str,str):
            idx = self._get_time(date_str)
            _temp = self.ds_icon['temp'][idx, :, :].values
            _pres = self.ds_icon['pres'][idx, :, :].values
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.temp[(i * self.nlayers) + z] = _temp[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]
                    self.pres[(i * self.nlayers) + z] = _pres[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]

            self.mesh.cell_data['theta'] = metpy.calc.potential_temperature(self.pres * units.Pa,
                                                                            self.temp * units.kelvin).magnitude

    def add_qv(self, date_str=None):
        if date_str is None:
            print("add_qv: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            _qv = self.ds_icon['qv'][idx, :, :].values
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.qv[(i * self.nlayers) + z] = _qv[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]
            self.mesh.cell_data['qv'] = self.qv

    def add_qc(self, date_str=None):
        if date_str is None:
            print("add_qc: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            _qc = self.ds_icon['qc'][idx, :, :].values
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.qc[(i * self.nlayers) + z] = _qc[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]
            self.mesh.cell_data['qc'] = self.qc

    def add_temp(self, date_str=None):
        if date_str is None:
            print("add_temp: Please provide a date")
            pass
        if isinstance(date_str, str):
            idx = self._get_time(date_str)
            # Get u and v
            _temp = self.ds_icon['temp'][idx, :, :].values
            for i in range(0, self.ncells, 1):
                for z in range(0, self.nlayers):
                    self.temp[(i * self.nlayers) + z] = _temp[(len(self.ds_icon.height) - self.nlayers - 1) - z, i]
            self.mesh.cell_data['temp'] = self.temp

    def add_ice(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str,str):
            idx = self._get_time(date_str)
            self.topo.cell_data['ice'] = self.ds_ext['ICE'][:].values

    def add_shfl(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str,str):
            idx = self._get_time(date_str)
            self.topo.cell_data['shfl_s'] = self.ds_icon['shfl_s'][idx, :].values

    def add_tsk(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str,str):
            idx = self._get_time(date_str)
            self.topo.cell_data['t_sk'] = self.ds_icon['t_sk'][idx, :].values

    def add_t2m(self, date_str=None):
        if date_str is None:
            print("add_tsk: Please provide a date")
            pass
        if isinstance(date_str,str):
            idx = self._get_time(date_str)
            self.topo.cell_data['t_2m'] = self.ds_icon['t_2m'][idx, 0, :].values

    def get_mesh(self):
        return self.mesh

    def get_vertices(self):
        return self.np_points

    def get_cells(self):
        return self.cells

    def get_cell_types(self):
        return self.cell_type

    def get_z_ifc(self):
        return self.z_ifc

    def get_z_mc(self):
        return self.z_mc

    def create_topo(self):

        np_points = np.ndarray([3 * self.ncells, 3])
        cells = np.ndarray([self.ncells, 4]).astype(int)
        cell_type = []

        # WGS84 zu UTM Zone 32N (Österreich liegt meist in Zone 32N)
        transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

        idx = 0
        for i in range(0, self.ncells, 1):
            for j in range(3):
                clon = self.clon_vertices[i, j]
                clat = self.clat_vertices[i,j]
                # Umrechnen
                x, y = transformer.transform(clon, clat)
                np_points[idx, 0] = np.array(x)
                np_points[idx, 1] = np.array(y)
                np_points[idx, 2] = np.array(0)
                idx = idx+1
            cells[i, :] = [3, idx - 3, idx - 2, idx - 1]
            cell_type = cell_type + [CellType.TRIANGLE]

        topo = pv.UnstructuredGrid(cells, cell_type, np_points)
        topo.cell_data['z_ifc'] = self.z_ifc[self.nfaces-1, :]

        # Remove duplicated
        ctopo = topo.clean()

        # Convert cell data to point data
        ptopo = ctopo.cell_data_to_point_data()

        # Assign height to coordinates
        for i in range(ptopo.points.shape[0]):
            ptopo.points[i, 2] = ptopo.point_data['z_ifc'][i]

        self.topo = ptopo

        return ptopo

    def create_grid(self):

        _z_mc = self.ds_icon.z_mc.values

        # WGS84 zu UTM Zone 32N (Österreich liegt meist in Zone 32N)
        transformer = Transformer.from_crs("epsg:4326", "epsg:32632", always_xy=True)

        for i in range(0, self.ncells, 1):
            for z in range(self.nfaces):
                for j in range(3):
                    x, y = transformer.transform(self.clon_vertices[i, j], self.clat_vertices[i, j])
                    _x = x
                    _y = y
                    _z = z
                    self._add_vertice(_x,_y,_z)
                if z > 0:
                    # Store first cell index
                    ids = self.idx - 6

                    # Create cell
                    self.cells[(i * self.nlayers) + (z - 1), :] = [24, 5,
                                                                   3, ids, ids + 1, ids + 2,
                                                                   3, ids + 3, ids + 4, ids + 5,
                                                                   4, ids, ids + 1, ids + 4, ids + 3,
                                                                   4, ids + 1, ids + 2, ids + 5, ids + 4,
                                                                   4, ids + 2, ids, ids + 3, ids + 5]

                    # Define cell type
                    self.cell_type[(i * self.nlayers) + (z - 1)] = CellType.POLYHEDRON

                    # Height information at cell center
                    self.z_mc[(i * self.nlayers) + (z - 1)] = _z_mc[self.height_full-z,i]

        # Create mnesh
        cmesh = pv.UnstructuredGrid(self.get_cells(), self.get_cell_types(), self.get_vertices())

        # Remove duplicated vertices
        cmesh = cmesh.clean()

        # Add height information as cell data
        cmesh.cell_data['z_mc'] = self.get_z_mc()
        pmesh = cmesh.cell_data_to_point_data()

        # Assign height to coordinates
        pmesh.points[:, 2] = pmesh.point_data['z_mc'][:]

        # for i in range(pmesh.points.shape[0]):
        #     if i < (self.topo.points.shape[0]):
        #         pmesh.points[i, 2] = self.topo.point_data['z_ifc'][i]
        #     else:
        #         pmesh.points[i, 2] = pmesh.point_data['z_mc'][i]

        self.mesh = pmesh

        return self.mesh


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

def main():

    '''
    # --  define path, file and variable name
    finit = '/data/projects/hefex/output/v3/lbc_ic_51m/init_ML_20230820T000000Z.nc'
    fext = '/data/projects/hefex/output/v3/exp_R3B15_51m/external_parameter_icon_hef_DOM01_tiles.nc'
    fgrid = '/data/projects/hefex/input/grids/local/hef_51m_DOM07.nc'
    ficon = '/data/projects/hefex/output/v3/exp_R3B15_51m/LES_DOM01_ML_0012.nc'
    '''

    # Create mesh
    imesh = icon_mesh(fgrid, ficon, fext, nlayers=50)
    imesh.print_time()
    date_str = '20230824 12:05:00'

    # Create orography file
    ptopo = imesh.create_topo()

    # Add 2D variables
    imesh.add_tsk(date_str)
    imesh.add_t2m(date_str)
    imesh.add_ice(date_str)
    imesh.add_shfl(date_str)
    ptopo.save('icon_topo-51m_surface.vtk')


    # Make a copy to avoid altering original
    #flat_mesh = ptopo.copy()

    # Set all Z coordinates to 0 to project onto XY plane
    #flat_mesh.points[:, 2] = 4000  # Project to z=0
    #flat_mesh.save('icon_topo-103m_flat.vtk')


    # Create 3D mesh
    cmesh = imesh.create_grid()

    # Add variables
    imesh.add_theta(date_str)
    #imesh.add_qv(date_str)
    #imesh.add_qc(date_str)
    #imesh.add_temp(date_str)
    #imesh.add_wind_vector(date_str)

    # Store mesh
    cmesh = imesh.get_mesh()
    cmesh.save('icon_mesh.vtk')

if __name__ == '__main__':
    main()
