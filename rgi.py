import geopandas as gpd

# Load the RGI6 shapefile
rgi_path = "../RGI2000-v7.0-G-11_central_europe.shp"
gdf = gpd.read_file(rgi_path)

# Preview the attribute table
print(gdf.head())

# Check column names to identify the glacier name field
print(gdf.columns)

# Filter by glacier name
hintereisferner = gdf[gdf['glac_name'].str.contains("Hintereisferner", case=False, na=False)]

# Display result
print(hintereisferner)