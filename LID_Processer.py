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


def parse_line(line, expected_fields=200):
    #### Options ####

    if line.startswith("VR"):
        header_match = re.match(r"^(VR\d{1,3})", line)
        if not header_match:
            raise ValueError("No valid header found")
        header = header_match.group(1)

        rest = line[len(header):]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 6]
            j = 0

            while i + 6 + j < len(rest) and not rest[i + 6 + j - 1].isspace() and not rest[i + 6 + j].isspace():
                chunk += rest[i + 6 + j]
                j += 1

            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    numbers = re.findall(r"-?\d{1,3}\.\d{2}", chunk)
                    for num in numbers:
                        fields.append(float(num))
                except ValueError:
                    fields.append(np.nan)
            i += 6 + j
            j = 0
    elif line.startswith("VVU"):
        header = "VVU"

        rest = line[len(header) + 0:]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 6]
            j = 0

            while i + 6 + j < len(rest) and not rest[i + 6 + j - 1].isspace() and not rest[i + 6 + j].isspace():
                chunk += rest[i + 6 + j]
                j += 1

            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    numbers = re.findall(r"-?\d{1,3}\.\d{2}", chunk)
                    for num in numbers:
                        fields.append(float(num))
                except ValueError:
                    fields.append(np.nan)
            i += 6 + j
            j = 0
    elif line.startswith("VVV"):
        header = "VVV"

        rest = line[len(header) + 0:]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 6]
            j = 0

            while i + 6 + j < len(rest) and not rest[i + 6 + j - 1].isspace() and not rest[i + 6 + j].isspace():
                chunk += rest[i + 6 + j]
                j += 1

            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    numbers = re.findall(r"-?\d{1,3}\.\d{2}", chunk)
                    for num in numbers:
                        fields.append(float(num))
                except ValueError:
                    fields.append(np.nan)
            i += 6 + j
            j = 0
    elif line.startswith("VVW"):
        header = "VVW"

        rest = line[len(header) + 0:]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 6]
            j = 0

            while i + 6 + j < len(rest) and not rest[i + 6 + j - 1].isspace() and not rest[i + 6 + j].isspace():
                chunk += rest[i + 6 + j]
                j += 1

            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    numbers = re.findall(r"-?\d{1,3}\.\d{2}", chunk)
                    for num in numbers:
                        fields.append(float(num))
                except ValueError:
                    fields.append(np.nan)
            i += 6 + j
            j = 0
    elif line.startswith("V "):
        header = "V"

        rest = line[len(header) + 2:]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 6]
            j = 0

            while i + 6 + j < len(rest) and not rest[i + 6 + j - 1].isspace() and not rest[i + 6 + j].isspace():
                chunk += rest[i + 6 + j]
                j += 1

            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    numbers = re.findall(r"\d{1,3}\.\d{2}", chunk)
                    for num in numbers:
                        fields.append(float(num))
                except ValueError:
                    fields.append(np.nan)
            i += 6 + j
            j = 0
    elif line.startswith("D"):
        header = "D"

        rest = line[len(header) + 3:]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 5]
            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    fields.append(float(chunk))
                except ValueError:
                    fields.append(np.nan)
            i += 6
    elif line.startswith("S W"):
        header = "S W"

        rest = line[len(header) + 2:]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 4]
            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    fields.append(float(chunk))
                except ValueError:
                    fields.append(np.nan)
            i += 6
    elif line.startswith("SN"):
        header_match = re.match(r"^(SN\d{1,3})", line)
        if not header_match:
            raise ValueError("No valid header found")
        header = header_match.group(1)

        rest = line[len(header):]

        fields = []
        i = 0
        while i < len(rest):
            chunk = rest[i:i + 6]
            j = 0

            while i + 6 + j < len(rest) and not rest[i + 6 + j - 1].isspace() and not rest[i + 6 + j].isspace():
                chunk += rest[i + 6 + j]
                j += 1

            if chunk.strip() == "":
                fields.append(np.nan)
            else:
                try:
                    if chunk.strip().lower() in {"inf", "+inf"}:
                        fields.append(np.inf)
                    elif chunk.strip().lower() == "-inf":
                        fields.append(-np.inf)
                    else:
                        numbers = re.findall(r"-?\d{1,3}\.\d{1}", chunk)
                        for num in numbers:
                            fields.append(float(num))
                except ValueError:
                    fields.append(np.nan)
            i += 6 + j
            j = 0
    elif line.startswith("ER"):
        # Extract header (ER + 1–3 digits)
        header_match = re.match(r"^(ER\d{1,3})", line)
        if not header_match:
            raise ValueError("No valid header found")
        header = header_match.group(1)

        rest = line[len(header):]

        fields = []
        for chunk in rest.split():
            chunk = chunk.strip()
            if chunk == "" or chunk == "0000" or chunk == "0":
                fields.append(np.nan)
            else:
                try:
                    fields.append(int(chunk))
                except ValueError:
                    fields.append(np.nan)

    else:
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


def parse_line_old(line, expected_fields=200):

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


def parse_lid_file(lid_path, exclude=(""), scale_height=1, debug=False):
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
                    header, fields = parse_line(line)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Failed to parse line: {line} | Error: {e}")
                    continue

                if len(fields) != 200:
                    if debug:
                        print(f"[DEBUG] Skipping line with {len(fields)} fields (expected 200): {header} - {line}")
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
            "height": heights * scale_height  # Apply scaling if needed
        }
    )

    return ds


def lid2xr(dir_path, exclude=("R ", "VR", "VN", "ER"), scale_height=1, debug=False):
    paths = list(Path(dir_path).rglob('*.lid'))

    ds_list = []
    for p in tqdm(paths, desc="Parsing LID files"):
        try:
            ds = parse_lid_file(p, exclude=exclude, scale_height=scale_height, debug=debug)
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


def plot_lidar_heatmap(ds, variable, save_as, cmap="viridis", vmax=None):
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
    plt.savefig(save_as)
    plt.show()


def main():
    lid_pathSt = r"C:\Users\malte\OneDrive\Handy\Stare"
    dsSt = lid2xr(lid_pathSt, exclude=("R "), scale_height=18, debug=True)
    dsSt.to_netcdf(r"C:\Users\malte\OneDrive\Handy\lidar_data_Stare.nc")

    # provide the VAD path
    lid_path35 = r"C:\Users\malte\OneDrive\Handy\VAD35"
    ds35 = lid2xr(lid_path35, exclude=("R "), scale_height=10.3, debug=True)
    ds35.to_netcdf(r"C:\Users\malte\OneDrive\Handy\lidar_data_VAD35.nc")

    '''
    plot_lidar_heatmap(ds35, "V", vmax=20, save_as="Velocity_Heatmap_35.png")
    plot_lidar_heatmap(ds35, "D", cmap="twilight", vmax=None, save_as="Direction_Heatmap_35.png")
    plot_lidar_heatmap(ds35, "SN1", vmax=None, save_as="SN1_Heatmap_35.png")
    '''

    lid_path70 = r"C:\Users\malte\OneDrive\Handy\VAD70"
    ds70 = lid2xr(lid_path70, exclude=("R "), scale_height=16.9, debug=True)
    ds70.to_netcdf(r"C:\Users\malte\OneDrive\Handy\lidar_data_VAD70.nc")

    '''
    plot_lidar_heatmap(ds70, "V", vmax=20, save_as="Velocity_Heatmap_70.png")
    plot_lidar_heatmap(ds70, "D", cmap="twilight", vmax=None, save_as="Direction_Heatmap_70.png")
    plot_lidar_heatmap(ds70, "SN1", vmax=None, save_as="SN1_Heatmap_70.png")
    '''

    dsSt = dsSt.assign_coords(type="Stare").expand_dims("type")
    ds35 = ds35.assign_coords(type="VAD35").expand_dims("type")
    ds70 = ds70.assign_coords(type="VAD70").expand_dims("type")

    ds = xr.merge([ds35, ds70], compat="no_conflicts")
    ds.to_netcdf(r"C:\Users\malte\OneDrive\Handy\lidar_data.nc")

if __name__ == "__main__":
    main()