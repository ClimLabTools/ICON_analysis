import geopandas as gpd
from shapely.geometry import mapping, Polygon
import pyvista as pv
import numpy as np
import triangle

def shapely_to_pv_surface(poly: Polygon):
    '''
    Function to convert shapely polygon geometry to pv.Polygon object for clipping the unstructured mesh
    '''

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


def vtk_shp_clip(vtk, shp, vis=False):
    '''
    Function to clip a VTK shp file. Can be used to limit the energy flux slices to the glaciated area
    '''
    polygon = shp.geometry.union_all()

    # Get pyvista surface for clipping
    surf = shapely_to_pv_surface(polygon)

    # Add height information to the surface to generate intersection with the icon mesh
    z_min, z_max = vtk.bounds[4], vtk.bounds[5]
    height = z_max - z_min
    extruded_surf = surf.extrude((0, 0, height * 1.1))

    # generate grid
    vtk_clip = vtk.clip_surface(extruded_surf, invert=True, progress_bar=True)

    # visualize if needed
    if vis:
        plotter = pv.Plotter()
        plotter.add_mesh(vtk_clip, color='lightblue', opacity=0.8, label='Clipped VTK')
        plotter.add_mesh(surf, color='red', line_width=3, label='Clip Polygon')
        plotter.add_legend()
        plotter.add_axes()
        plotter.show()

    return vtk_clip


def main():
    vtk = pv.read("../../Data/HU_D/icon_data_40.vtk", progress_bar=True)

    shp = gpd.read_file("../../Data/RGI2000-v7.0-G-11_central_europe.shp")
    HEF = shp[shp['glac_name'].str.contains("Hintereisferner", case=False, na=False)]

    vtk_clip = vtk_shp_clip(vtk, HEF, vis=True)


if __name__ == "__main__":
    main()