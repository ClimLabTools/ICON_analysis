import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import re
from pathlib import Path
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from tempfile import mkdtemp
import shutil, os

from click import progressbar
from tqdm import tqdm
from joblib import Parallel, delayed

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


def parse_lid_file(
    lid_path,
    exclude=("R", "VR", "VN", "ER"),   # fix: empty-string default was too broad
    scale_height=1,
    debug=False,
):
    time_stamps = []
    heights = None

    # header -> list aligned with time steps; each element is either list[float] or None
    variables = {}
    seen_this_time = set()
    current_time_active = False

    def _finalize_time_step():
        for h, seq in variables.items():
            if h not in seen_this_time:
                seq.append(None)
        seen_this_time.clear()

    # Stream lines; no whole-file read
    with open(lid_path, "r", encoding="latin1", buffering=1024*1024) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue

            if line.startswith("LID"):
                if current_time_active:
                    _finalize_time_step()
                parts = line.split()
                try:
                    raw_time = parts[1]
                    time_stamps.append(
                        f"20{raw_time[:2]}-{raw_time[2:4]}-{raw_time[4:6]} "
                        f"{raw_time[6:8]}:{raw_time[8:10]}:{raw_time[10:12]}"
                    )
                    current_time_active = True
                except Exception:
                    current_time_active = False
                continue

            if line.startswith("H "):
                try:
                    # much faster than regex loop
                    heights = np.fromstring(line[3:], sep=" ", dtype=float)
                except Exception as e:
                    if debug:
                        print(f"[DEBUG] Failed to parse heights: {e}")
                    heights = None
                continue

            # Skip excluded prefixes or numeric-leading lines
            if line.startswith(exclude) or line[0].isdigit():
                continue
            if not current_time_active:
                continue

            try:
                header, fields = parse_line(line)  # your function
            except Exception as e:
                if debug:
                    print(f"[DEBUG] Failed to parse line: {line} | Error: {e}")
                continue

            seq = variables.get(header)
            if seq is None:
                # backfill earlier times with None
                variables[header] = seq = [None] * (len(time_stamps) - 1)
            seq.append(fields)
            seen_this_time.add(header)

    if current_time_active:
        _finalize_time_step()

    if heights is None or len(time_stamps) == 0:
        raise ValueError(f"Missing heights or timestamps in file {lid_path}")

    times = pd.to_datetime(time_stamps)
    H = len(heights)
    data_vars = {}
    for header, seq in variables.items():
        if len(seq) < len(times):
            seq.extend([None] * (len(times) - len(seq)))
        rows = []
        for fields in seq:
            if fields is None:
                rows.append(np.full(H, np.nan, dtype=float))
            else:
                arr = np.asarray(fields, dtype=float)
                if arr.size != H:
                    if arr.size > H:
                        arr = arr[:H]
                    else:
                        tmp = np.full(H, np.nan, dtype=float)
                        tmp[:arr.size] = arr
                        arr = tmp
                rows.append(arr)
        data = np.vstack(rows)
        data_vars[header.strip()] = (("time", "height"), data)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": times, "height": heights * scale_height},
    )
    return ds


def _worker_parse_to_temp_zarr(idx, path, exclude, scale_height, dtype, chunks, tmp_root, debug):
    try:
        ds = parse_lid_file(path, exclude=exclude, scale_height=scale_height, debug=debug)
        if dtype:
            for v in ds.data_vars:
                if np.issubdtype(ds[v].dtype, np.floating):
                    ds[v] = ds[v].astype(dtype)
        if chunks:
            ds = ds.chunk(chunks if isinstance(chunks, dict) else {"time": chunks})  # requires dask

        tmp_dir = mkdtemp(prefix=f"lidpart_{idx:06d}_", dir=tmp_root)
        tmp_store = os.path.join(tmp_dir, "part.zarr")
        # ---- write each part as Zarr v2 ----
        ds.to_zarr(tmp_store, mode="w", zarr_format=2)
        return (idx, tmp_store, int(ds.sizes["time"]))
    except Exception as e:
        if debug:
            print(f"[DEBUG] Worker failed on {Path(path).name}: {e}")
        return (idx, "", -1)

def lid2xr_parallel_streaming(
    dir_path,
    out_zarr,
    n_workers=None,
    exclude=("R", "VR", "VN", "ER"),
    scale_height=1.0,
    dtype="float32",
    chunks={"time": 4096},
    encoding=None,            # compressor dict; applied on FIRST write only
    buffer_factor=4,
    debug=False,
):
    paths = sorted(Path(dir_path).rglob("*.lid"))
    if not paths:
        raise FileNotFoundError(f"No .lid files under {dir_path}")

    out_zarr = str(out_zarr)
    if Path(out_zarr).exists():
        shutil.rmtree(out_zarr)

    # avoid BLAS oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    tmp_root = mkdtemp(prefix="lid_tmproot_")
    created = False
    next_expected = 0
    buffer = {}
    max_buffer = max(4, (n_workers or 1) * buffer_factor)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(_worker_parse_to_temp_zarr, i, str(p), exclude, scale_height,
                      dtype, chunks, tmp_root, debug)
            for i, p in enumerate(paths)
        ]

        for fut in as_completed(futs):
            idx, tmp_store, n_time = fut.result()
            if n_time < 0:
                continue
            buffer[idx] = tmp_store

            # flush in order; keep buffer bounded
            while (next_expected in buffer) or (len(buffer) >= max_buffer):
                if next_expected not in buffer:
                    break

                tmp_store = buffer.pop(next_expected)

                if not created:
                    # first piece defines schema / encoding
                    ds0 = xr.open_zarr(tmp_store, consolidated=False, zarr_format=2)
                    ds0.to_zarr(out_zarr, mode="w", encoding=encoding, zarr_format=2)
                    del ds0
                    created = True
                else:
                    # subsequent pieces: open and APPEND along time
                    ds_part = xr.open_zarr(tmp_store, consolidated=False, zarr_format=2)
                    ds_part.to_zarr(out_zarr, mode="a", append_dim="time")
                    del ds_part

                # remove temp store to free disk
                shutil.rmtree(Path(tmp_store).parent, ignore_errors=True)
                next_expected += 1

    shutil.rmtree(tmp_root, ignore_errors=True)
    if not created:
        raise ValueError("No valid datasets parsed.")

    # optional: consolidate metadata for faster later opens
    try:
        import zarr
        zarr.consolidate_metadata(out_zarr)
        return xr.open_zarr(out_zarr, consolidated=True, zarr_format=2)
    except Exception:
        # fall back if zarr not available
        return xr.open_zarr(out_zarr, consolidated=False, zarr_format=2)

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
    lid_pathSt = r"E:\HU_DATA\Streamline_conv\Userfile 3"

    ds = lid2xr_parallel_streaming(
        dir_path=lid_pathSt,
        out_zarr=r"E:\HU_DATA\Streamline_conv\user3.zarr",
        n_workers=2,  # ~ number of physical cores
        exclude=("R", "VR", "ER", "SN"), # "R", "VR", "VN", "ER", "SN"
        scale_height=18,
        dtype="float32",
        chunks={"time": 4096},
        debug=True,
    )

if __name__ == "__main__":
    main()

