import pyvista as pv
import geopandas as gpd
from shapely.geometry import Point
import glob
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

# --- 1. Load the shapefile (polygon)
gdf = gpd.read_file("test.shp")
polygon = gdf.geometry.union_all()  # merge multiple features if needed

# 1. Get all filenames (e.g., all .vtk files in a folder)
filepaths = glob.glob("./icon_2023*.vtk")  # adjust the pattern as needed

# Create empty DataFrame
df = pd.DataFrame(columns=['date', 'theta'])

# 2. Define the slicing plane (e.g., z-slice)
z_height = 2500.0
origin = (0, 0, z_height)
normal = (0, 0, 1)

# loop over all vtk files
for filepath in filepaths:
    # --- Load the mesh
    mesh = pv.read(filepath)
    # --- create slice
    slice_plane = mesh.slice(normal=normal, origin=origin)

    # --- 3. Filter slice points that lie inside the polygon
    points = slice_plane.points[:, :2]  # drop Z (assumed flat)
    mask = np.array([polygon.contains(Point(p)) for p in points])

    # --- 4. Extract the subset using the shape file
    subset = slice_plane.extract_points(mask)

    # Optional: save vtk slice
    subset.save(f"slice_{filepath.split('/')[-1]}")

    # === After closing: compute final statistics
    if subset and 'theta' in subset.cell_data:
        values = subset.cell_data['theta']
        values = values[~np.isnan(values)]  # Clean NaNs

        print("\n=== Final Statistics over all selected regions ===")
        print(f"  Mean     : {np.mean(values)}")
        print(f"  Variance : {np.var(values)}")
        print(f"  Min      : {np.min(values)}")
        print(f"  Max      : {np.max(values)}")

