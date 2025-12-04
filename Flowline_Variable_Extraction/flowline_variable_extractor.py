import os
import glob
import numpy
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
from tqdm.auto import tqdm
from matplotlib.colors import ListedColormap



def get_intersection_cell_ids(flowline, grid_path):
    grid = gpd.read_file(grid_path, mask=flowline)
    return [int(id) for id in grid["cell_id"]], grid

def get_icon_cell_values(ICON_PATH, cell_ids, variables: list[str], heights: list[int], times = []):
    ds = xr.open_dataset(ICON_PATH)

    if not times:
        times = ds.time.values

    ds_select = ds.sel(time=times, ncells_2=cell_ids, height=heights)[variables]#.sortby("ncells_2")

    return(ds_select)

def map_xarray2gpd(grid, ds, times=[]):
    if not times:
        times = ds['time'].values

    results = {}

    for t in times:
        gdf_copy = grid.copy()
        gdf_copy = gdf_copy#.sort_values(by="cell_id")

        for h in ds["height"].values:
            dh = ds.sel(height=h)
            for var in dh.data_vars:
                da = dh[var]

                da_t = da.sel(time=t)

                vals = da_t.values
                cells = da_t['ncells_2'].values if 'ncells_2' in da_t.coords else da['ncells_2'].values

                mapping = pd.Series(data=vals, index=cells)

                h_suffix = f"h_{int(h)}"
                newvar = f"{h_suffix}_{var}"
                gdf_copy[newvar] = gdf_copy.index.map(mapping)

            if ("u" in ds.data_vars) and ("v" in ds.data_vars):
                gdf_copy[f"h_{int(h)}_wsp"] = np.sqrt(gdf_copy[f"h_{int(h)}_u"]**2 + gdf_copy[f"h_{int(h)}_v"]**2)

        results[str(t)] = gdf_copy

    return results

def compute_h(rho_0, rho_h, u):
    delta_rho = numpy.absolute(rho_h-rho_0)
    rho_mean = np.mean(np.concatenate((rho_h, rho_0), axis=0))
    return 0.25 * (rho_mean / 9.81) * (u**2 / delta_rho)

def map_h_to_xr(ds, low, high):
    udiff = ds.sel(height=low)["u"] - ds.sel(height=high)["u"]
    vdiff = ds.sel(height=low)["v"] - ds.sel(height=high)["v"]
    u = np.sqrt(udiff**2 + vdiff**2)
    return compute_h(rho_0=ds.sel(height=low)["rho"], rho_h=ds.sel(height=high)["rho"], u=u)

def yyyymmdd_fraction_to_datetime64_compact(t):
    tv = float(t)
    iv = int(tv)
    base = pd.to_datetime(f"{iv:08d}", format="%Y%m%d")
    frac_day = tv - iv
    seconds = frac_day * 86400.0
    ns = int(round(seconds * 1e9))
    ts = base + pd.to_timedelta(ns, unit="ns")
    return pd.to_datetime(ts).to_datetime64()

def iterate_through_sims(root, flowline_path, grid_path, heights=[147,150]):
    fl_df = pd.read_csv(flowline_path)
    flp_gdf = gpd.GeoDataFrame(fl_df, geometry=gpd.points_from_xy(fl_df.lon, fl_df.lat), crs="EPSG:4326")
    line = LineString(flp_gdf.geometry.tolist())
    fll_gdf = gpd.GeoDataFrame({'geometry':[line],
                                "line_dist":[fl_df["Distance along flowline"].tolist()],
                                "surface_height":[fl_df["surface_height"].tolist()],
                                "slope_deg":[fl_df["slope_deg"].tolist()]}, crs=flp_gdf.crs)

    g, grid = get_intersection_cell_ids(fll_gdf, grid_path)
    gdf = grid#.sort_values(by="cell_id")
    if gdf.crs is not None and getattr(gdf.crs, "to_epsg", lambda: None)() != 3857:
        gdf = gdf.to_crs(epsg=3857)
    centroids_proj = gdf.geometry.centroid
    centroids = gpd.GeoSeries(centroids_proj, crs="EPSG:3857").to_crs(epsg=4326)
    lons = centroids.x.to_numpy()
    lats = centroids.y.to_numpy()
    n_points = len(gdf)

    all_da_list = []
    all_rho_list = []
    all_u_list = []
    all_v_list = []
    all_temp_list = []
    global_time_list = []

    pattern = f"{root}/LES_51m_ml*.nc"
    varname = "h"

    for icon_file in tqdm(glob.glob(pattern)[:]):
        if icon_file.endswith("0121.nc"):
            print(f"Skipping {icon_file}")
            continue

        ds_sel = get_icon_cell_values(icon_file, g, variables=["rho","u","v","temp"], heights=heights)

        for t in ds_sel.time.values:
            ds_t = ds_sel.sel(time=t)
            h = map_h_to_xr(ds_t, heights[0], heights[1])
            h_arr = np.asarray(h)

            if h_arr.ndim == 1:
                data = h_arr[np.newaxis, :]
            elif h_arr.ndim == 0:
                data = h_arr[np.newaxis, np.newaxis]
            else:
                data = h_arr

            t64 = yyyymmdd_fraction_to_datetime64_compact(t)
            da_h = xr.DataArray(
                data=data,
                dims=("time","point"),
                coords={"time":[t64], "point": np.arange(n_points)},
                name="h"
            )
            all_da_list.append(da_h)
            
            for var in ["rho", "u", "v", "temp"]:
                da_var = ds_t[var].expand_dims("time")  
                da_var = da_var.assign_coords(time=[t64])
                
                if "ncells_2" in da_var.dims:
                    da_var = da_var.rename({"ncells_2": "point"})
                    da_var = da_var.assign_coords(point=np.arange(n_points))  
            
                if var == "rho":
                    all_rho_list.append(da_var)
                elif var == "u":
                    all_u_list.append(da_var)
                elif var == "temp":
                    all_temp_list.append(da_var)
                else:
                    all_v_list.append(da_var)
        
            global_time_list.append(t64)

    final_xr = xr.Dataset(
        {
            "h": xr.concat(all_da_list, dim="time"), # .sortby("time")
            "rho": xr.concat(all_rho_list, dim="time"),
            "u": xr.concat(all_u_list, dim="time"),
            "v": xr.concat(all_v_list, dim="time"),
            "temp": xr.concat(all_temp_list, dim="time"),
        },
        coords={
            "time": global_time_list,
            "point": np.arange(n_points),
            "lat": lats,
            "lon": lons
        }
    )


    return final_xr, global_time_list



def downward_mask(point_gdf, grid_gdf, wind_vec, degree_deviation_treshold=45, degree_deviation_treshold_2=None):
    point_gdf_proj = point_gdf.to_crs(epsg=3857)

    point_gdf_proj["cx"] = point_gdf_proj.geometry.x
    point_gdf_proj["cy"] = point_gdf_proj.geometry.y

    point_gdf_proj["cx_next"] = point_gdf_proj["cx"].shift(-1)
    point_gdf_proj["cy_next"] = point_gdf_proj["cy"].shift(-1)

    point_gdf_proj["dx"] = point_gdf_proj["cx_next"] - point_gdf_proj["cx"]
    point_gdf_proj["dy"] = point_gdf_proj["cy_next"] - point_gdf_proj["cy"]

    point_gdf_proj["dist_to_next_m"] = np.hypot(point_gdf_proj["dx"], point_gdf_proj["dy"])

    vec_components = point_gdf_proj[["dx", "dy"]].iloc[:-1].to_numpy()
    dist_list = point_gdf_proj["dist_to_next_m"].iloc[:-1].to_numpy()

    dx = vec_components[:, 0]
    dy = vec_components[:, 1]

    bearings = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    point_gdf_proj["bearings"] = np.append(bearings, np.zeros(1))

    bearings_icon = (np.degrees(np.arctan2(wind_vec[:,0], wind_vec[:,1])) + 360) % 360

    grid_proj = grid_gdf.to_crs(point_gdf_proj.crs)

    joined_grid = gpd.sjoin_nearest(grid_proj, point_gdf_proj[['bearings', 'geometry']],
                               how='left', distance_col='dist')

    joined_grid["bearings_icon"] = bearings_icon

    abs_diff = np.abs(joined_grid["bearings"] - joined_grid["bearings_icon"])
    abs_diff = np.minimum(abs_diff, 360 - abs_diff)
    joined_grid["diffs"] = abs_diff
    if degree_deviation_treshold is not None:
        joined_grid["bearings_mask"] = (abs_diff > degree_deviation_treshold).astype(int) * 1 + (abs_diff > degree_deviation_treshold_2).astype(int) * 10
    else:
        joined_grid["bearings_mask"] = (abs_diff > degree_deviation_treshold).astype(int)
    joined_grid["bearings_mask"] = np.where(joined_grid["bearings_mask"] == 10, np.nan, joined_grid["bearings_mask"])


    return joined_grid



############### MAIN ###############

def main():
    FLOWLINE_PATH = r"C:\Users\malte\OneDrive\Arbeit_HU\Tasks\2025_T12_Theory_Test\hef_flowline_slope.csv"
    GRID_VECTOR_PATH = r"C:\Users\malte\OneDrive\Arbeit_HU\Tasks\2025_T12_Theory_Test\grid_51m.gpkg"
    TEST_ICON_PATH = r"C:\Users\malte\OneDrive\Arbeit_HU\Tasks\2025_T12_Theory_Test\LES_51m_ml_0041.nc"

    fl_df = pd.read_csv(FLOWLINE_PATH)
    flp_gdf = gpd.GeoDataFrame(fl_df, geometry=gpd.points_from_xy(fl_df.lon, fl_df.lat), crs="EPSG:4326")
    line = LineString(flp_gdf.geometry.tolist())
    fll_gdf = gpd.GeoDataFrame({'geometry': [line],
                                "line_dist": [fl_df["Distance along flowline"].tolist()],
                                "surface_height": [fl_df["surface_height"].tolist()],
                                "slope_deg": [fl_df["slope_deg"].tolist()]}, crs=flp_gdf.crs)

    g, grid = get_intersection_cell_ids(fll_gdf, GRID_VECTOR_PATH)

    icon_nc = xr.open_dataset(r"C:\Users\malte\OneDrive\Arbeit_HU\Tasks\2025_T12_Theory_Test\flowline_h.nc")

    '''
    closest_to = "2025-08-15 14:30:00"
    targ = np.datetime64(pd.Timestamp(closest_to))
    t_pos = int(np.argmin(np.abs(icon_nc['time'].values - targ)))

    time_label = icon_nc['time'].values[t_pos]
    print("closest time coordinate:", pd.to_datetime(time_label))

    u = icon_nc.isel(time=t_pos, height=1)["u"].squeeze()
    v = icon_nc.isel(time=t_pos, height=1)["v"].squeeze()
    h = icon_nc.isel(time=t_pos)["h"].squeeze()

    wind_vec = np.vstack((u.values, v.values)).T

    gdf_new = downward_mask(point_gdf=flp_gdf, grid_gdf=grid, wind_vec=wind_vec, degree_deviation_treshold=135, degree_deviation_treshold_2=45)


    img = gdf_new.plot(
        column="bearings",
        cmap="twilight",
        vmin=0, vmax=360,
        legend=True)
    plt.show()

    gdf_new["h"] = h

    img = gdf_new.plot(
        column="h",
        legend=True,
        vmax=20
    )
    plt.show()

    img = gdf_new.plot(
        column="diffs",
        legend=True,
        #vmax=20
    )
    plt.show()

    colors = ['#0072B2', '#D55E00']
    cmapbin = ListedColormap(colors)

    img = gdf_new.plot(
        column="bearings_mask",
        cmap=cmapbin,
        vmin=0, vmax=1,
        legend=True)
    plt.show()

    '''

    #all_h = []
    #all_bear = []

    fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
    #for t, a in tqdm(zip(range(0, 24), ax.flatten())):
    for d, a in tqdm(zip(range(5, 30), ax.flatten())):
        all_h = []
        all_bear = []
        for t in range(0, 24):
            if t < 10:
                closest_to = f"2025-08-{d} 0{t}:00:00"
            else:
                closest_to = f"2025-08-{d} {t}:00:00"
            targ = np.datetime64(pd.Timestamp(closest_to))
            t_pos = int(np.argmin(np.abs(icon_nc['time'].values - targ)))

            time_label = icon_nc['time'].values[t_pos]
            #print("closest time coordinate:", pd.to_datetime(time_label))

            u = icon_nc.isel(time=t_pos, height=1)["u"].squeeze()
            v = icon_nc.isel(time=t_pos, height=1)["v"].squeeze()
            h = icon_nc.isel(time=t_pos)["h"].squeeze()

            wind_vec = np.vstack((u.values, v.values)).T

            gdf_new = downward_mask(point_gdf=flp_gdf, grid_gdf=grid, wind_vec=wind_vec, degree_deviation_treshold=135)

            gdf_new["h"] = h

            [all_h.append(hs) for hs in gdf_new.h.values]

            #img = gdf_new.plot(
            #    ax = a,
            #    column="h",
            #    legend=True,
            #    vmax = 20,
            #)
            #a.set_title(time_label)

        #plt.tight_layout()
        #plt.savefig("h_15_TS.png")
        #plt.show()

    #fig, ax = plt.subplots(nrows=6, ncols=4, figsize=(20, 20))
    #for t, a in tqdm(zip(range(0, 24), ax.flatten())):
    #for d in tqdm(range(6, 31)):
        for t in range(0, 24):
            if t < 10:
                closest_to = f"2025-08-{d} 0{t}:00:00"
            else:
                closest_to = f"2025-08-{d} {t}:00:00"
            targ = np.datetime64(pd.Timestamp(closest_to))
            t_pos = int(np.argmin(np.abs(icon_nc['time'].values - targ)))

            time_label = icon_nc['time'].values[t_pos]
            #print("closest time coordinate:", pd.to_datetime(time_label))

            u = icon_nc.isel(time=t_pos, height=1)["u"].squeeze()
            v = icon_nc.isel(time=t_pos, height=1)["v"].squeeze()
            h = icon_nc.isel(time=t_pos)["h"].squeeze()

            wind_vec = np.vstack((u.values, v.values)).T

            gdf_new = downward_mask(point_gdf=flp_gdf, grid_gdf=grid, wind_vec=wind_vec, degree_deviation_treshold=135)

            [all_bear.append(bear) for bear in gdf_new.bearings_mask.values]

            #colors = ['#0072B2', '#D55E00']
            #cmapbin = ListedColormap(colors)

            #img = gdf_new.plot(
            #    ax = a,
            #    column="bearings_mask",
            #    cmap=cmapbin,
            #    vmin=0, vmax=1,
            #    legend=True,
            #)
            #a.set_title(time_label)

        #plt.tight_layout()
        #plt.savefig("bear_15_TS.png")
        #plt.show()

        g0 = [v for b, v in zip(all_bear, all_h) if b == 0]
        g1 = [v for b, v in zip(all_bear, all_h) if b == 1]

        if len(g1) < 0.01 * len(grid) * 24:
            print(f"Case: {time_label} with length {len(g1)} is smaller than f{0.01 * len(grid) * 24}")

        print(f"Case: {time_label} with length {len(g1)} compared to {len(grid) * 24} possible cases")

        data = [list(g0), list(g1)]
        #fig, ax = plt.subplots(figsize=(7, 5))

        a.boxplot(data, labels=["0", "1"], showfliers=False)
        a.set_xlabel("Wind In Agreement")
        a.set_ylabel("h")

        medians = [np.median(d) if len(d) > 0 else np.nan for d in data]
        for i, m in enumerate(medians, start=1):
            if np.isfinite(m):
                a.text(i, m, f"{m:.3g}", ha="center", va="bottom")
        a.set_title(time_label)

    plt.tight_layout()
    fig.savefig(f"Boxplot_Wind_per_day.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()