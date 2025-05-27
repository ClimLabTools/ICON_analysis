import geopandas as gpd
from shapely.geometry import mapping, Polygon
import pyvista as pv
import numpy as np
import triangle
from collections import defaultdict

def shapely_to_pv_surface(poly: Polygon):

    poly = poly.buffer(0)

    coords = np.array(poly.exterior.coords[:-1])
    n_pts = len(coords)

    # rebuild vertices and edges of the polygon
    vertices = coords[:, :2]
    segments = [[i, i + 1] for i in range(n_pts - 1)] + [[n_pts - 1, 0]]

    # dict for triangulation
    poly_dict = {
        'vertices': vertices.astype(np.float64),
        'segments': segments
    }

    # use graph based triangulation to rebuild polygon
    try:
        tri = triangle.triangulate(poly_dict, 'p')
    except Exception as e:
        print("Triangulation failed:", e)
        raise

    if 'triangles' not in tri:
        raise RuntimeError("Triangulation didn't produce triangles.")

    # build pyvista surface mesh from triangulated polygon
    faces = np.hstack([[3, *tri['triangles'][i]] for i in range(len(tri['triangles']))])
    z_flat = np.full(tri['vertices'].shape[0], 0)
    points_3d = np.column_stack((tri['vertices'], z_flat))
    mesh = pv.PolyData(points_3d, faces)

    return mesh

def extract_ground_surface(mesh: pv.UnstructuredGrid, normal_threshold: float = -0.1, height_threshold: float = 0.95):
    surface = mesh.extract_surface()

    surface.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    normals = surface['Normals']
    down_mask = normals[:, 2] < normal_threshold

    cell_centers = surface.cell_centers()
    z_coords = cell_centers.points[:, 2]

    z_min = z_coords.min()
    z_max = z_coords.max()
    max_z = z_min + height_threshold * (z_max - z_min)

    elevation_mask = z_coords < max_z

    final_mask = down_mask & elevation_mask

    ground_surface = surface.extract_cells(np.where(final_mask)[0])

    return ground_surface

def vtk_shp_clip(vtk, shp, return_feature, vis=False, notebook=False, normal_threshold: float = -0.1, height_threshold: float = 0.95):
    '''
    :param return_feature: Either 'surface' or '3D' depending on whether you want the full 3D mesh-structure clipped
                           or just the ground/surface mesh.
    '''

    polygon = shp.geometry.union_all()

    # Get pyvista surface for clipping
    surf = shapely_to_pv_surface(polygon)

    # Add height information to the surface to generate intersection with the icon mesh
    z_min, z_max = vtk.bounds[4], vtk.bounds[5]
    height = z_max - z_min
    extruded_surf = surf.extrude((0, 0, height * 1.1), capping=True)

    # generate grid
    vtk_clip = vtk.clip_surface(extruded_surf, invert=True, progress_bar=True)

    threeD = ["3D", "3d"]
    twoD = ["surface", "Surface", "2D", "2d", "ground", "Ground"]
    if return_feature not in threeD + twoD:
        raise ValueError(f"Invalid argument: '{return_feature}'. Must be one of the elements in {threeD} or {twoD}.")
    elif return_feature in twoD:
        vtk_clip = extract_ground_surface(vtk_clip, normal_threshold = normal_threshold, height_threshold = height_threshold)

    # visualize if needed
    if vis:
        if notebook:
            plotter = pv.Plotter(notebook=True)
        else:
            plotter = pv.Plotter()
        
        plotter.add_mesh(vtk_clip, scalars=vtk_clip.array_names[0], cmap="viridis", opacity=0.8, label='Clipped VTK')
        plotter.add_mesh(surf, color='red', line_width=3, label='Clip Polygon')
        plotter.add_legend()
        plotter.add_axes()
        plotter.show()

    return vtk_clip


def main():
    vtk = pv.read("../../Data/HU_D/icon_data_40.vtk", progress_bar=True)

    shp = gpd.read_file("../../Data/RGI2000-v7.0-G-11_central_europe.shp")
    HEF = shp[shp['glac_name'].str.contains("Hintereisferner", case=False, na=False)]

    vtk_clip = vtk_shp_clip(vtk, HEF, return_feature="2D", vis=True)


if __name__ == "__main__":
    main()