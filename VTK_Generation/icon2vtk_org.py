import time, os
import xarray as xr

import numpy as np
import metpy
from metpy.units import units

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# from   matplotlib.collections import PolyCollection

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

import pyvista as pv
from pyvista import CellType


def load_file(fname):
    ds = xr.open_dataset(fname)
    return ds

def icon_get_grid(ds_grid):
    # -- get coordinates and convert radians to degrees
    clon = np.rad2deg(ds_grid.clon.values)
    clat = np.rad2deg(ds_grid.clat.values)
    clon_vertices = np.rad2deg(ds_grid.clon_vertices.values)
    clat_vertices = np.rad2deg(ds_grid.clat_vertices.values)

    ncells, nv = clon_vertices.shape[0], clon_vertices.shape[1]

    # -- create the triangles
    clon_vertices = np.where(clon_vertices < -180., clon_vertices + 360., clon_vertices)
    clon_vertices = np.where(clon_vertices > 180., clon_vertices - 360., clon_vertices)

    triangles = np.zeros((ncells, nv, 3), np.float32)

    for i in range(0, ncells, 1):
        triangles[i, :, 0] = np.array(clon_vertices[i, :])
        triangles[i, :, 1] = np.array(clat_vertices[i, :])
        triangles[i, :, 2] = np.array(0)

    # -- create polygon/triangle collection
    # coll = PolyCollection(triangles, transform=ccrs.Geodetic())

    return triangles


class icon_mesh():

    def __init__(self, ds_grid, ds_icon, nlayers):
        self.clon_vertices = np.rad2deg(ds_grid.clon_vertices.values)
        self.clat_vertices = np.rad2deg(ds_grid.clat_vertices.values)
        self.ncells, nv = self.clon_vertices.shape[0], self.clon_vertices.shape[1]

        self.height = ds_icon.sizes['height_3']
        self.height_full = ds_icon.sizes['height']
        self.nlayers = nlayers
        self.nfaces = self.nlayers + 1
        self.np_points = np.ndarray([3 * (self.ncells * self.nfaces), nv])
        self.cell_type = np.ndarray(self.ncells * self.nlayers).astype(object)
        self.cells = np.ndarray([self.ncells * self.nlayers, 25]).astype(int)
        self.z_ifc = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.w = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.u = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.v = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.temp = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.pres = np.ndarray([self.ncells * self.nlayers]).astype(float)
        self.idx = 0

        self.ds = ds_icon
        self.ds_z_ifc = ds_icon.z_ifc.values

    def _add_vertice(self, _lon, _lat, _z):
        self.np_points[self.idx, 0] = _lon
        self.np_points[self.idx, 1] = _lat
        self.np_points[self.idx, 2] = _z
        self.idx += 1

    def face_to_center(self, _var, t=0):
        _w = self.ds[_var][t, :, :].values
        for i in range(0, self.ncells, 1):
            for z in range(1, self.nfaces):
                self.w[(i * self.nlayers) + (z - 1)] = (_w[self.height - z - 1, i] + _w[self.height - z, i]) / 2

    def add_wind_vector(self, t=0):
        _u = self.ds['u'][t, :, :].values
        _v = self.ds['v'][t, :, :].values
        for i in range(0, self.ncells, 1):
            for z in range(0, self.nlayers):
                self.u[(i * self.nlayers) + z] = _u[(self.height_full - 1) - z, i]
                self.v[(i * self.nlayers) + z] = _v[(self.height_full - 1) - z, i]
        self.mesh.cell_data['w'] = self.w
        self.mesh.cell_data['u'] = self.u
        self.mesh.cell_data['v'] = self.v
        self.mesh.cell_data['wind'] = np.column_stack((self.u, self.v, self.w))

    def add_theta(self, t=0):
        print(t)
        _temp = self.ds['temp'][t, :, :].values
        _pres = self.ds['pres'][t, :, :].values
        for i in range(0, self.ncells, 1):
            for z in range(0, self.nlayers):
                self.temp[(i * self.nlayers) + z] = _temp[(self.height_full - 1) - z, i]
                self.pres[(i * self.nlayers) + z] = _pres[(self.height_full - 1) - z, i]

        self.mesh.cell_data['theta'] = metpy.calc.potential_temperature(self.pres * units.Pa,
                                                                        self.temp * units.kelvin).magnitude

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

    def create_grid(self):
        for i in range(0, self.ncells, 1):
            for z in range(self.nfaces):
                for j in range(3):
                    self._add_vertice(self.clon_vertices[i, j], self.clat_vertices[i, j], z)

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
                    self.z_ifc[(i * self.nlayers) + (z - 1)] = self.ds_z_ifc[self.height - z, i] + \
                                                               (self.ds_z_ifc[self.height - z, i - 1] - self.ds_z_ifc[
                                                                   self.height - z, i]) / 2.0

        # Create mnesh
        cmesh = pv.UnstructuredGrid(self.get_cells(), self.get_cell_types(), self.get_vertices())

        # Remove duplicated vertices
        cmesh = cmesh.clean()

        # Add height information as cell data
        cmesh.cell_data['z_ifc'] = imesh.get_z_ifc()
        cmesh = cmesh.cell_data_to_point_data()

        # Assign height to coordinates
        for i in range(cmesh.points.shape[0]):
            cmesh.points[i, 2] = cmesh.point_data['z_ifc'][i] / 50000

        self.mesh = cmesh

        return cmesh

    def create_topo(self):

        np_points = np.ndarray([3 * self.ncells, 3])
        cells = np.ndarray([self.ncells, 4]).astype(int)
        cell_type = []

        idx = 0
        for i in range(0, self.ncells, 1):
            for j in range(3):
                np_points[idx, 0] = np.array(self.clon_vertices[i,j])
                np_points[idx, 1] = np.array(self.clat_vertices[i, j])
                np_points[idx, 2] = np.array(0)
                idx = idx+1
            cells[i, :] = [3, idx - 3, idx - 2, idx - 1]
            cell_type = cell_type + [CellType.TRIANGLE]

        topo = pv.UnstructuredGrid(cells, cell_type, np_points)
        topo.cell_data['z_ifc'] = self.ds_z_ifc[self.height_full, :]
        # Remove duplicated
        ctopo = topo.clean()

        # Convert cell data to point data
        pdata = ctopo.cell_data_to_point_data()

        # Assign height to coordinates
        for i in range(pdata.points.shape[0]):
            pdata.points[i, 2] = pdata.point_data['z_ifc'][i] / 50000

        return pdata

# --  define path, file and variable name
finit = '/data/projects/hefex/output/v3/lbc_ic_51m/init_ML_20230819T000000Z.nc'
fgrid = '/data/projects/hefex/input/grids/local/hef_51m_DOM07.nc'
ficon = '/data/projects/hefex/output/v3/exp_R3B15_51m/LES_DOM01_ML_0010.nc'

# Load grid and extPar file
ds_grid = load_file(fgrid)
ds_init = load_file(finit)
ds_icon = load_file(ficon)

imesh = icon_mesh(ds_grid, ds_icon, nlayers=10)
cmesh = imesh.create_grid()
imesh.add_theta()
imesh.add_wind_vector()
cmesh = imesh.get_mesh()
pmesh = cmesh.cell_data_to_point_data()
print(cmesh.cell_data.keys())

ptopo = imesh.create_topo()

ptopo.save('icon_topo.vtk')
pmesh.save('icon_data.vtk')
