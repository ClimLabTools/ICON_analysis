import pyvista as pv
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

def vtk_to_shapefile(input_vtk, output_shp):
    # Load the VTK mesh
    mesh = pv.read(input_vtk)

    geoms = []

    # Handle points
    if mesh.n_cells == 0 and mesh.n_points > 0:
        for pt in mesh.points:
            geoms.append(Point(pt[:2]))  # Use only X, Y for 2D
    else:
        for i in range(mesh.n_cells):
            cell = mesh.extract_cells(i)
            pts = cell.points[:, :2]  # Take X, Y only

            if len(pts) == 1:
                geoms.append(Point(pts[0]))
            elif len(pts) == 2:
                geoms.append(LineString(pts))
            elif len(pts) > 2:
                geoms.append(Polygon(pts))

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:32632")  # Set your CRS here

    # Write to shapefile
    gdf.to_file(output_shp)

    print(f"Saved shapefile to: {output_shp}")

# Example usage
vtk_to_shapefile("slice_20230819T00:00:00.vtk", "output.shp")