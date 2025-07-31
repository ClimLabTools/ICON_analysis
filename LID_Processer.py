import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

global counter
counter = [0, 0, 0]

global differences
differences = []


def parse_fixed_width_line(line, expected_fields=200):

    counter[2] += 1

    # Find start of data (first digit or 5 spaces after first space)
    match = re.search(r'\s[-+]?\d| {5}', line)
    if not match:
        raise ValueError("No data start found in line.")

    start_idx = match.start() + 1
    header = line[:start_idx].strip()
    rest = line[start_idx:]

    # Regex to match either exactly 5 spaces (one NaN) or a float number
    value_pattern = re.compile(r'( {5})|([-+]?\d+(?:\.\d{1,2})?)')
    raw_matches = value_pattern.findall(rest)

    fields = []
    for space_match, num_match in raw_matches:
        if space_match == '     ':
            fields.append(np.nan)
        elif num_match:
            try:
                fields.append(float(num_match))
            except ValueError:
                fields.append(np.nan)

    # Ensure fixed number of fields
    if len(fields) != expected_fields:
        differences.append(len(fields) - expected_fields)
    if len(fields) < expected_fields:
        counter[0] += 1
        fields += [np.nan] * (expected_fields - len(fields))
    elif len(fields) > expected_fields:
        counter[1] += 1
        fields = fields[:expected_fields]

    return header, fields


def parse_lid_file(lid_path, exclude=(""), debug=False):
    time_stamps = []
    heights = None
    records_per_time = []

    with open(lid_path, "r", encoding="latin1") as f:
        lines = f.readlines()

    current_time = None
    current_record = None

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue

        if line.startswith("LID"):
            parts = line.split()
            try:
                raw_time = parts[1]
                time_str = f"20{raw_time[:2]}-{raw_time[2:4]}-{raw_time[4:6]} {raw_time[6:8]}:{raw_time[8:10]}:{raw_time[10:12]}"
                current_time = time_str
                time_stamps.append(current_time)
                current_record = {}
                records_per_time.append(current_record)
            except Exception:
                current_time = None
                current_record = None
        elif line.startswith("H "):
            heights = np.array([float(v) for v in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', line[3:])])
        elif line.startswith(exclude) or line[0].isdigit():
            continue
        else:
            if current_record is not None:
                try:
                    header, fields = parse_fixed_width_line(line)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Failed to parse line: {line} | Error: {e}")
                    continue

                if len(fields) != 200:
                    if debug:
                        print(f"[DEBUG] Skipping line with {len(fields)} fields (expected 200): {header}")
                    continue
                current_record[header] = fields

    if heights is None or len(time_stamps) == 0:
        raise ValueError(f"Missing heights or timestamps in file {lid_path}")

    # Collect all unique headers
    all_headers = sorted({key for rec in records_per_time for key in rec.keys()})
    variables = {header: [] for header in all_headers}

    for rec in records_per_time:
        for header in all_headers:
            fields = rec.get(header, [np.nan] * 200)
            if len(fields) != len(heights):
                if debug:
                    print(f"[DEBUG] Adjusting field length for '{header}' from {len(fields)} to {len(heights)}")
                fields = fields[:len(heights)] + [np.nan] * (len(heights) - len(fields))
            variables[header].append(fields)

    # Convert times to datetime
    times = pd.to_datetime(time_stamps)

    # Build xarray dataset
    data_vars = {}
    for header, records in variables.items():
        arr = np.array(records, dtype=float)

        if arr.shape != (len(times), len(heights)):
            if debug:
                print(f"[DEBUG] Skipping variable '{header}': shape {arr.shape} != ({len(times)}, {len(heights)})")
            continue

        data_vars[header.strip()] = (("time", "height"), arr)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": times,
            "height": heights * 18  # Apply scaling if needed
        }
    )

    return ds


def lid2xr(dir_path, exclude=("R ", "VR", "VN", "ER"), debug=False):
    paths = list(Path(dir_path).rglob('*.lid'))

    ds_list = []
    for p in tqdm(paths, desc="Parsing LID files"):
        try:
            ds = parse_lid_file(p, exclude=exclude, debug=debug)
            if ds is not None:
                ds_list.append(ds)
        except Exception as e:
            if debug:
                print(f"[DEBUG] Skipping {p.name}: {e}")
            continue

    if not ds_list:
        raise ValueError("No valid datasets parsed from directory")

    combined_ds = xr.concat(ds_list, dim="time").sortby("time")
    return combined_ds


def plot_lidar_heatmap(ds, variable, cmap="viridis", vmax=None):
    time_nums = mdates.date2num(ds["time"].values)

    data = ds[variable].values.T
    if vmax is None:
        vmax = np.nanmax(data)

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmax=vmax, interpolation="none",
                   extent=[time_nums[0], time_nums[-1], ds["height"].values[-1], ds["height"].values[0]])

    ax.invert_yaxis()

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    fig.autofmt_xdate(rotation=45)

    ax.set_xlabel("Time")
    ax.set_ylabel("Height (m)")
    ax.set_title(f"{variable} Heatmap (Height vs Time)")
    fig.colorbar(im, label=variable)

    plt.tight_layout()
    plt.show()


def main():

    # provide the VAD
    lid_path = r"C:\Users\malte\OneDrive\Handy\VAD"

    ds = lid2xr(lid_path, exclude=("R ", "VR", "VN", "ER", "S"), debug=True)
    ds.to_netcdf(r"C:\Users\malte\OneDrive\Handy\lidar_data.nc")

    plot_lidar_heatmap(ds, "V", vmax=20)
    plot_lidar_heatmap(ds, "D", cmap="twilight", vmax=None)
    #plot_lidar_heatmap(ds, "SN1", vmax=None)

    print(counter)
    plt.hist(differences, bins=50, edgecolor="black")
    plt.show()


if __name__ == "__main__":
    main()