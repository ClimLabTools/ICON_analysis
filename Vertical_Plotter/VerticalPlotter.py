import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
import pyvista as pv
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from pyproj import Transformer
import os
import matplotlib.cm as cm


# Further Steps
    # Adapt height to the new icon2vtk

    # Add functionality for spline inputs


class VerticalPlotter:
    '''
    Class to generate vertical plots from icon vtk files
    Recquires:
    1. vtk file,
    2. vector file of the transect line,
    3. name of the variable to be plotted,
    4. maximum height of the transect as a list of [maximum array height, maximum plot height]
    '''
    def __init__(self, icon_vtk, epsg, gdf_line, plot_variable, max_height, grid_width=1, interp_method='linear'):
        self.icon_vtk = icon_vtk.cell_data_to_point_data()
        self.epsg = epsg
        self.grid_width = grid_width
        self.interp_method = interp_method
        self.plot_variable = plot_variable
        self.max_height = max_height
        self.gdf_line = gdf_line
        self.vector_transform_option = 1
        self.pv_line = None
        self.slice = None
        self.loni = None
        self.zi = None
        self.lon = None
        self.z = None
        self.values = None
        self.grid_values = None
        self.grid_values_x = None
        self.grid_values_y = None
        self.grid_values_z = None
        self.grid_values_lon = None
        self.grid_values_z_new = None
        self.grid_values_perp = None
        self.offsets = None

        self.scalar = True if len(self.icon_vtk.get_array(self.plot_variable).shape) == 1 else False

    def plotter_info(self):
        print(self.icon_vtk)

    def adadpt_z(self, col_new='z_ifc'):
        '''
        Used to transform the z-values of the normalized vtk z-coords to the true meters stored in the z_ifc column
        :param col_new: The column name containing the correct z-values in meters
        '''

        col_old = 'old_z_vals'
        new_z = np.array(self.icon_vtk.get_array(col_new), dtype=self.icon_vtk.points.dtype)
        old_z = self.icon_vtk.points[:, 2]
        points = self.icon_vtk.points.copy()
        points[:, 2] = new_z
        self.icon_vtk.points = points

        self.icon_vtk[col_old] = old_z

        self.icon_vtk = self.icon_vtk.threshold((0, self.max_height[0]), scalars="z_ifc")

        self.icon_vtk = pv.UnstructuredGrid(self.icon_vtk)

    def gpd_line_2_pv_line(self, n_points):
        '''
        Takes the gdf line and transforms it into a pv spline to be compatible with the pv.slice_along_line function
        :param n_points: Choose how many points the spline should have.
        '''
        if self.gdf_line.crs != self.epsg:
            self.gdf_line = self.gdf_line.to_crs(self.epsg)

        print(f"Line length: {self.gdf_line.length}")

        coords = np.array(self.gdf_line.geometry.values[0].coords)
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

        self.pv_line = pv.Spline(coords, n_points)

    def generate_icon_slice(self):
        '''
        Generates a 2D vertical slice along the defined pv spline through the vtk.
        Note that the function always seems to extend the slice beyond the bounds of the spline until it reaches the boundaries of the vtk.
        Thus, the file result needs to be filtered to contain only the relevant data.
        '''

        x_bounds = (self.pv_line.points[:, 0].min(), self.pv_line.points[:, 0].max())
        y_bounds = (self.pv_line.points[:, 1].min(), self.pv_line.points[:, 1].max())
        z_bounds = (self.icon_vtk.points[:, 2].min(), self.icon_vtk.points[:, 2].max())

        all_points = self.icon_vtk.points
        mask = (
                (all_points[:, 0] >= x_bounds[0]) & (all_points[:, 0] <= x_bounds[1]) &
                (all_points[:, 1] >= y_bounds[0]) & (all_points[:, 1] <= y_bounds[1]) &
                (all_points[:, 2] >= z_bounds[0]) & (all_points[:, 2] <= z_bounds[1])
        )

        clipped_vtk = self.icon_vtk.extract_points(mask, include_cells=True)

        self.slice = clipped_vtk.slice_along_line(self.pv_line, progress_bar=True)


    def interpolate_icon_slice(self):
        '''
        Interpolates the slice according to the chosen interpolation method and stores the resulting arrays and arrays coordinates in the class.
        '''


        if self.scalar:
            concat_array = np.concatenate((self.slice.points, self.slice.get_array(self.plot_variable)[:, None]), axis=1)
            sorted_indices = np.argsort(concat_array[:, 0])
            sorted_points = self.slice.points[sorted_indices]
            x, y, z = sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2]

            if self.epsg == "EPSG:4326":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
                x, y = transformer.transform(x, y)

            x = np.array(x)
            y = np.array(y)

            # Compute consecutive distances
            dsts = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
            line_length = np.sum(dsts)
            lon = np.concatenate([[0], np.cumsum(dsts)])

            values = self.slice.get_array(self.plot_variable)[sorted_indices]

            loni = np.arange(lon.min(), lon.max(), self.grid_width)
            zi = np.arange(z.min(), z.max(), self.grid_width)
            loni, zi = np.meshgrid(loni, zi)

            grid_values = griddata((lon, z), values, (loni, zi), method=self.interp_method)

            self.grid_values = grid_values
            self.loni = loni
            self.zi = zi
            self.lon = lon
            self.z = z
            self.values = values

        else:
            concat_array = np.concatenate((self.slice.points, self.slice.get_array(self.plot_variable)), axis=1)
            sorted_indices = np.argsort(concat_array[:, 0])
            sorted_points = self.slice.points[sorted_indices]
            x, y, z = sorted_points[:, 0], sorted_points[:, 1], sorted_points[:, 2]

            if self.epsg == "EPSG:4326":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
                x, y = transformer.transform(x, y)

            x = np.array(x)
            y = np.array(y)

            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
            line_length = np.sqrt(dx ** 2 + dy ** 2)
            t_x = dx / line_length
            t_y = dy / line_length
            lon = (x - x[0]) * t_x + (y - y[0]) * t_y


            values = self.slice.get_array(self.plot_variable)[sorted_indices]

            loni = np.arange(lon.min(), lon.max(), self.grid_width)
            zi = np.arange(z.min(), z.max(), self.grid_width)
            loni, zi = np.meshgrid(loni, zi)

            grid_values_x = griddata((lon, z), values[:,0], (loni, zi), method=self.interp_method)
            grid_values_y = griddata((lon, z), values[:,1], (loni, zi), method=self.interp_method)
            grid_values_z = griddata((lon, z), values[:,2], (loni, zi), method=self.interp_method)

            d = np.array([x[-1] - x[0], y[-1] - y[0], 0])
            z_axis = np.array([0, 0, 1])
            n = np.cross(z_axis, d)

            def vector2proj(us, n):
                n_norm = np.linalg.norm(n)
                dot_prods = np.dot(us, n)
                offset_vectors = (dot_prods[:, np.newaxis]) * n / n_norm ** 2
                us_proj = us - offset_vectors
                s_dists = dot_prods / n_norm

                return us_proj, s_dists, offset_vectors

            proj_v, signed_offsets, _ = vector2proj(us=values, n=n)


            grid_offsets = griddata((lon, z), signed_offsets, (loni, zi), method=self.interp_method)

            u_lon = proj_v[:,0] * t_x + proj_v[:,1] * t_y

            grid_values_lon = griddata((lon, z), u_lon, (loni, zi), method=self.interp_method)
            grid_values_z_new = griddata((lon, z), values[:,2], (loni, zi), method=self.interp_method)

            self.grid_values_lon = grid_values_lon
            self.grid_values_z_new = grid_values_z_new
            self.grid_values_x = grid_values_x
            self.grid_values_y = grid_values_y
            self.grid_values_z = grid_values_z
            self.loni = loni
            self.zi = zi
            self.lon = lon
            self.z = z
            self.values = values
            self.offsets = grid_offsets


    def clip_result_array(self):
        '''
        Used to filter for only the valid data points. Takes away the artifacts of the slice generation. Sets terrain
        and too high points to nan.
        '''
        lon_min, lon_max = self.lon.min(), self.lon.max()
        z_min, z_max = self.z.min(), self.z.max()
        lon_norm = (self.lon - lon_min) / (lon_max - lon_min)
        z_norm = (self.z - z_min) / (z_max - z_min)
        loni_norm = (self.loni - lon_min) / (lon_max - lon_min)
        zi_norm = (self.zi - z_min) / (z_max - z_min)

        tree = cKDTree(np.column_stack((lon_norm, z_norm)))
        distances, _ = tree.query(np.column_stack((loni_norm.ravel(), zi_norm.ravel())))

        threshold = 1

        if self.scalar:
            self.grid_values[distances.reshape(self.grid_values.shape) > threshold] = np.nan
        else:
            self.grid_values_x[distances.reshape(self.grid_values_x.shape) > threshold] = np.nan
            self.grid_values_y[distances.reshape(self.grid_values_y.shape) > threshold] = np.nan
            self.grid_values_z[distances.reshape(self.grid_values_z.shape) > threshold] = np.nan
            self.grid_values_lon[distances.reshape(self.grid_values_lon.shape) > threshold] = np.nan
            self.grid_values_z_new[distances.reshape(self.grid_values_z_new.shape) > threshold] = np.nan
            self.offsets[distances.reshape(self.offsets.shape) > threshold] = np.nan


    def return_interpolation_result(self):
        '''
        Function to return the interpolated results if desired.
        :return:
        '''

        result = {
            'extent': [self.lon.min(), self.lon.max(), self.z.min(), self.z.max()]
        }

        if self.scalar:
            result['type'] = 'scalar'
            result['grid_values'] = self.grid_values
        else:
            result['type'] = 'vector'
            result['grid_values'] = {
                'x': self.grid_values_x,
                'y': self.grid_values_y,
                'z': self.grid_values_z,
                'lon': self.grid_values_lon,
                'z_new': self.grid_values_z_new,
                'offset': self.offsets
            }
            result['interpolation_grid'] = {
                'lon': self.loni,
                'z': self.zi
            }

        return result

    def vertical_profile_plot(self, plot_type='standard', nan_color="black",
                              cmap_name='viridis', label=None, discrete=True,
                              contour=True, bins=10, c_lines=10, c_color='black',
                              scale=100, density=2, save_as='test', offset=True):
        '''
        Function to automatically generate vertical plots. Options are standard for scalar data and quiver, streamplot and standard for vector data.
        Standard for vector generates a three-element plot displaying the x, y and z components just as scalar data.
        :param plot_type: Choose between standard and quiver, streamplot.
        :param nan_color: Color for nan values. Important for color of the terrain.
        :param cmap_name: Name of the desired colormap.
        :param label: Label to be displayed on the variable name.
        :param discrete: Discretize the colormap. Default is True.
        :param contour: Add contour lines to the plot. Default is True. Currently only works for standard plots
        :param bins: Number of bins for the discrete colormap.
        :param c_lines: Levels for the contour lines. Either integer or list of desired levels. Default is 10
        :param c_color: Color for the contour lines. Default is black.
        :param scale: Scale for the quiver plot. Default is 100.
        :param density: Density for the streamplot. Default is 2.
        :param save_as: Save under this variable name.
        '''

        extent = [self.lon.min(), self.lon.max(), self.z.min(), self.z.max()]

        cmap = plt.get_cmap(cmap_name, bins).copy()

        if label == None:
            label = self.plot_variable

        if self.scalar:
            vmin, vmax = np.nanmin(self.grid_values), np.nanmax(self.grid_values)
            if discrete:
                colors = cmap(np.linspace(0, 1, bins))
                new_cmap = mcolors.ListedColormap(colors)
                new_cmap.set_bad(color=nan_color)
            else:
                new_cmap = cmap.copy()
                new_cmap.set_bad(color=nan_color)

            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))

            img1 = ax.imshow(self.grid_values, origin='lower', extent=extent, cmap=new_cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(img1, ax=ax, shrink=0.5, label=label)

            if contour:
                contour = ax.contour(self.grid_values, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                ax.clabel(contour, inline=True, fontsize=10, fmt="%.0f")

            ax.set_xlabel("Line Distance (m)")
            ax.set_ylabel("Altitude (m)")
            ax.set_aspect('equal')
            ax.set_title(f"Interpolated Vertical Plot Displaying {label}")
            ax.set_xlim([min(self.lon), max(self.lon)])
            ax.set_ylim([min(self.z), self.max_height[1]])

            plt.tight_layout()

            plt.savefig(f"{save_as}_Scalar_Standard.png", dpi=300, bbox_inches="tight")
            plt.show()

        else:

            if plot_type == 'quiver':
                X = self.loni
                Y = self.zi
                U = self.grid_values_lon
                V = self.grid_values_z_new
                magnitude = np.sqrt(U ** 2 + V ** 2)

                step = 50
                X_sub = X[::step, ::step]
                Y_sub = Y[::step, ::step]
                U_sub = U[::step, ::step]
                V_sub = V[::step, ::step]
                magnitude_sub = magnitude[::step, ::step]

                print(X_sub.shape, Y_sub.shape, U_sub.shape, V_sub.shape, self.offsets[::step, ::step].shape)

                if offset:
                    vmin, vmax = np.nanmin(self.offsets[::step, ::step]), np.nanmax(self.offsets[::step, ::step])
                else:
                    vmin, vmax = np.nanmin(magnitude_sub), np.nanmax(magnitude_sub)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                if discrete:
                    colors = cmap(np.linspace(0, 1, bins))
                    new_cmap = mcolors.ListedColormap(colors)
                    new_cmap.set_bad(color=nan_color)
                else:
                    new_cmap = cmap.copy()
                    new_cmap.set_bad(color=nan_color)


                # Background
                bg = np.ones_like(self.grid_values_x)
                bg[np.isnan(self.grid_values_x)] = 0

                plt.figure(figsize=(10, 6))
                plt.imshow(bg, origin='lower', extent=extent, cmap="gray")
                if offset:
                    Q = plt.quiver(X_sub, Y_sub, U_sub, V_sub, self.offsets[::step, ::step], cmap=new_cmap, scale=scale,
                                   pivot='middle', norm=norm)
                else:
                    Q = plt.quiver(X_sub, Y_sub, U_sub, V_sub, magnitude_sub, cmap=new_cmap, scale=scale, pivot='middle', norm=norm)
                cbar = plt.colorbar(Q, ax=plt.gca(), shrink=0.5, label=label)
                cbar.set_label(label)
                plt.xlabel("Line Distance (m)")
                plt.ylabel("Altitude (m)")
                plt.title(f"Interpolated Vertical Quiver Plot Displaying {label}")
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.ylim([min(self.z), self.max_height[1]])
                plt.savefig(f"{save_as}_Quiver.png", dpi=300, bbox_inches="tight")
                plt.show()

            elif plot_type == 'streamplot':
                X = self.loni
                Y = self.zi
                U = self.grid_values_lon
                V = self.grid_values_z_new
                magnitude = np.sqrt(U ** 2 + V ** 2)

                step = 50
                X_sub = X[::step, ::step]
                Y_sub = Y[::step, ::step]
                U_sub = U[::step, ::step]
                V_sub = V[::step, ::step]
                magnitude_sub = magnitude[::step, ::step]

                if offset:
                    vmin, vmax = np.nanmin(self.offsets[::step, ::step]), np.nanmax(self.offsets[::step, ::step])
                else:
                    vmin, vmax = np.nanmin(magnitude_sub), np.nanmax(magnitude_sub)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

                if discrete:
                    colors = cmap(np.linspace(0, 1, bins))
                    new_cmap = mcolors.ListedColormap(colors)
                    new_cmap.set_bad(color=nan_color)
                else:
                    new_cmap = cmap.copy()
                    new_cmap.set_bad(color=nan_color)

                # Background
                bg = np.ones_like(self.grid_values_x)
                bg[np.isnan(self.grid_values_x)] = 0

                plt.figure(figsize=(10, 6))
                plt.imshow(bg, origin='lower', extent=extent, cmap="gray")
                if offset:
                    plt.streamplot(X_sub, Y_sub, U_sub, V_sub, density=density, linewidth=1, color=self.offsets[::step, ::step],
                                   cmap=new_cmap, norm=norm)
                else:
                    plt.streamplot(X_sub, Y_sub, U_sub, V_sub, density=density, linewidth=1, color=magnitude_sub, cmap=new_cmap, norm=norm)
                sm = cm.ScalarMappable(cmap=new_cmap, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=plt.gca(), shrink=0.5, label=label)

                plt.xlabel("Line Distance (m)")
                plt.ylabel("Altitude (m)")
                plt.title(f"Interpolated Vertical Streamplot Displaying {label}")
                plt.gca().set_aspect('equal')
                plt.ylim([min(self.z), self.max_height[1]])
                plt.savefig(f"{save_as}_Streamplot.png", dpi=300, bbox_inches="tight")
                plt.show()

            else:

                if discrete:
                    colors = cmap(np.linspace(0, 1, bins))
                    new_cmap = mcolors.ListedColormap(colors)
                    new_cmap.set_bad(color=nan_color)
                else:
                    new_cmap = cmap.copy()
                    new_cmap.set_bad(color=nan_color)

                fig, ax = plt.subplots(ncols=1, nrows=3, figsize=(10, 18))

                img1 = ax[0].imshow(self.grid_values_x, origin='lower', extent=extent, cmap=new_cmap, vmin=np.nanmin(self.grid_values_x), vmax=np.nanmax(self.grid_values_x))
                fig.colorbar(img1, ax=ax[0], shrink=0.5, label=f'{label}_x_component')

                if contour:
                    contour = ax[0].contour(self.grid_values_x, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                    ax[0].clabel(contour, inline=True, fontsize=10, fmt="%.0fm/s")


                img2 = ax[1].imshow(self.grid_values_y, origin='lower', extent=extent, cmap=new_cmap, vmin=np.nanmin(self.grid_values_y), vmax=np.nanmax(self.grid_values_y))
                fig.colorbar(img2, ax=ax[1], shrink=0.5, label=f'{label}_y_component')

                if contour:
                    contour = ax[1].contour(self.grid_values_y, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                    ax[1].clabel(contour, inline=True, fontsize=10, fmt="%.0fm/s")


                img3 = ax[2].imshow(self.grid_values_z, origin='lower', extent=extent, cmap=new_cmap, vmin=np.nanmin(self.grid_values_z), vmax=np.nanmax(self.grid_values_z))
                fig.colorbar(img3, ax=ax[2], shrink=0.5, label=f'{label}_z_component')

                if contour:
                    contour = ax[2].contour(self.grid_values_z, levels=c_lines, colors=c_color, linewidths=1, extent=extent)
                    ax[2].clabel(contour, inline=True, fontsize=10, fmt="%.0fm/s")

                components = ["x", "y", "z"]
                for axis, component in zip(ax.flatten(), components):
                    axis.set_xlabel("Line Distance (m)")
                    axis.set_ylabel("Altitude (m)")
                    axis.set_aspect('equal')
                    axis.set_title(f"Interpolated Vertical Plot Displaying {label} {component}-Component")
                    axis.set_xlim([min(self.lon), max(self.lon)])
                    axis.set_ylim([min(self.z), self.max_height[1]])

                plt.tight_layout()

                plt.savefig(f"{save_as}_Vector_Standard.png", dpi=300, bbox_inches="tight")
                plt.show()

    def pv_3d_visualization(self):
        '''
        Creates a pyvista 3D visualization of the line and slice. Currently, it has some issues with the z-position due to rescaling to and from meters.
        '''
        vtk_copy = self.icon_vtk.copy()
        slice_copy = self.slice.copy()
        line_copy = self.pv_line.copy()
        line_points = line_copy.points.copy()
        line_points[:,2] = 0
        line_copy.points = line_points

        vtk_z_min = np.min(vtk_copy.points[:,2])
        vtk_z_max = np.max(vtk_copy.points[:,2])
        slice_points = slice_copy.points.copy()
        slice_min_z = np.min(slice_points[:,2])
        slice_max_z = np.max(slice_points[:,2])
        scale = (vtk_z_max - vtk_z_min) / (slice_max_z - slice_min_z)
        shift = vtk_z_min - slice_min_z * scale
        slice_points[:, 2] = scale * slice_points[:, 2] + shift
        slice_copy.points = slice_points

        plotter = pv.Plotter()
        plotter.add_mesh(vtk_copy.outline(), color="black")
        plotter.add_mesh(slice_copy, scalars=self.plot_variable, cmap="viridis", show_scalar_bar=True, show_edges=True) # Slice is still not at the perfect z position but works for now
        plotter.add_mesh(line_copy, color="red", line_width=5)
        #plotter.add_mesh(vtk_copy, scalars=self.plot_variable, cmap="viridis", show_scalar_bar=True, show_edges=True)
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.show()

    def full_run(self, old_vtk=False, height_col_name='z_ifc', number_line_points=1000, autoplot=True, return_result=False, plot_3D=False,
                 plot_type='standard', nan_color="black",
                 cmap_name='viridis', label=None, discrete=True,
                 contour=True, bins=10, c_lines=10, c_color='black',
                 scale=100, density=2, save_as='test'):

        '''
        Performes a full run of the vertical profile plot workflow and generate the result arrays. Return and automatically plots are optional.
        :param height_col_name: Name of the vtk array name containing height information in meters.
        :param number_line_points: Number of horizontal segments along the line.
        :param autoplot: If true, plots are automatically generated.
        :param return_result: If true, the interpolated result arrays are returned. Receiving variable need to be set according to the variable type (scalar or vector).
        :param plot_3D: If true, a pyvista 3D visualization is generated. Default is False.
        :param plot_type: Choose between standard and quiver, streamplot.
        :param nan_color: Color for nan values. Important for color of the terrain.
        :param cmap_name: Name of the desired colormap.
        :param label: Label to be displayed on the variable name.
        :param discrete: Discretize the colormap. Default is True.
        :param contour: Add contour lines to the plot. Default is True. Currently only works for standard plots
        :param bins: Number of bins for the discrete colormap.
        :param c_lines: Levels for the contour lines. Either integer or list of desired levels. Default is 10
        :param c_color: Color for the contour lines. Default is black.
        :param scale: Scale for the quiver plot. Default is 100.
        :param density: Density for the streamplot. Default is 2.
        :param save_as: Save under this variable name.
        :return:
        '''

        if old_vtk:
            self.adadpt_z(col_new=height_col_name)
        self.gpd_line_2_pv_line(number_line_points)
        self.generate_icon_slice()
        self.interpolate_icon_slice()
        self.clip_result_array()
        if plot_3D:
            self.pv_3d_visualization()
        if autoplot:
            self.vertical_profile_plot(plot_type=plot_type, nan_color=nan_color,
                                       cmap_name=cmap_name, label=label, discrete=discrete,
                                       contour=contour, bins=bins, c_lines=c_lines, c_color=c_color,
                                       scale=scale, density=density, save_as=save_as)
        if return_result:
            return self.return_interpolation_result()




def main():

    icon_vtk_1 = pv.read(r"..\Flowspline_Data\icon_mesh.vtk")
    icon_vtk_2 = pv.read(r"..\Flowspline_Data\icon_mesh_theta_v.vtk")
    gdf_line = gpd.read_file(r"..\Flowspline_Data\hef_flowline.shp")

    print(icon_vtk_2.array_names)
    #print(icon_vtk.points[:5,:])

    VP_1 = VerticalPlotter(icon_vtk_1, "EPSG:32632", gdf_line, 'theta', max_height=[4000, 3800], grid_width=1, interp_method='linear')
    _ = VP_1.full_run(plot_type='standard', number_line_points=1000, save_as="Theta_Spline", plot_3D=True)

    VP_2 = VerticalPlotter(icon_vtk_2, "EPSG:32632", gdf_line, 'theta_v', max_height=[4000, 3800], grid_width=1, interp_method='linear')
    _ = VP_2.full_run(plot_type='standard', number_line_points=1000, save_as="Theta_v_Spline", plot_3D=True)



if __name__ == "__main__":
    main()
