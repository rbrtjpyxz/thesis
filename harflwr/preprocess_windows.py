import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from harflwr.data_precomputed import inter_subject_split_client_ids

# Claude used in this file for cleaning up and organizing
# also used for verification of logic
CHANNEL_CONFIGS = (
    "raw",
    "raw_mag",
    "raw_mag_deriv",
    "raw_mag_deriv_grav",
    "raw_grav",
    "mag_std_grav",
    "std_grav",
    "mag_std_grav_mag",
    "std_grav_mag",
    "mag_std_grav_xz",
    "raw_mag_deriv_grav_xz",
    "mag_std_grav_mag_xz",
    "mag",
    "deriv",
    "mag_deriv",
    "mag_std",
    "mag_deriv_std",
    "std",
)


def save_json(obj, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    f = open(path, "w")
    json.dump(obj, f, indent=2)
    f.close()


def list_client_files(data_dir):
    files = []
    for f in os.listdir(data_dir):
        if f.endswith(".csv"):
            files.append(os.path.join(data_dir, f))
    return sorted(files)


def compute_derived_columns(df, channel_config, window_seconds):
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    df = df.copy()

    needs_mag = channel_config in (
        "raw_mag", "raw_mag_deriv", "raw_mag_deriv_grav", "mag", "deriv",
        "mag_deriv", "mag_std", "mag_deriv_std", "mag_std_grav", "std_grav",
        "mag_std_grav_mag", "std_grav_mag", "mag_std_grav_xz",
        "raw_mag_deriv_grav_xz", "mag_std_grav_mag_xz", "std"
    )

    if needs_mag:
        df["acc_mag"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)

    needs_deriv = channel_config in (
        "raw_mag_deriv", "raw_mag_deriv_grav", "raw_mag_deriv_grav_xz",
        "deriv", "mag_deriv", "mag_deriv_std", "std"
    )

    if needs_deriv:
        dt = 1.0 / 20.0
        df["acc_dmag"] = (
            df.groupby("segment_id")["acc_mag"]
            .transform(lambda x: x.diff().fillna(0.0) / dt)
        )

    needs_std = channel_config in (
        "mag_std", "mag_deriv_std", "mag_std_grav", "std_grav",
        "mag_std_grav_mag", "std_grav_mag", "mag_std_grav_xz",
        "mag_std_grav_mag_xz", "std"
    )

    needs_grav_mag = channel_config in (
        "mag_std_grav_mag", "std_grav_mag", "mag_std_grav_mag_xz"
    )

    if needs_grav_mag:
        df["grav_mag"] = np.sqrt(df["grav_x"] ** 2 + df["grav_y"] ** 2 + df["grav_z"] ** 2)

    if needs_std:
        window_size = window_seconds * 20
        std_window = max(1, int(round(window_size * 0.2)))
        df["acc_mag_std"] = (
            df.groupby("segment_id")["acc_mag"]
            .transform(lambda x: x.rolling(window=std_window, min_periods=1).std().fillna(0.0))
        )

    if channel_config == "raw":
        sensor_cols = acc_cols
    elif channel_config == "raw_mag":
        sensor_cols = acc_cols + ["acc_mag"]
    elif channel_config == "raw_mag_deriv":
        sensor_cols = acc_cols + ["acc_mag", "acc_dmag"]
    elif channel_config == "mag":
        sensor_cols = ["acc_mag"]
    elif channel_config == "deriv":
        sensor_cols = ["acc_dmag"]
    elif channel_config == "mag_deriv":
        sensor_cols = ["acc_mag", "acc_dmag"]
    elif channel_config == "mag_std":
        sensor_cols = ["acc_mag", "acc_mag_std"]
    elif channel_config == "mag_deriv_std":
        sensor_cols = ["acc_mag", "acc_dmag", "acc_mag_std"]
    elif channel_config == "std":
        sensor_cols = ["acc_mag_std"]
    elif channel_config == "mag_std_grav":
        sensor_cols = ["acc_mag", "acc_mag_std", "grav_x", "grav_y", "grav_z"]
    elif channel_config == "std_grav":
        sensor_cols = ["acc_mag_std", "grav_x", "grav_y", "grav_z"]
    elif channel_config == "mag_std_grav_mag":
        sensor_cols = ["acc_mag", "acc_mag_std", "grav_mag"]
    elif channel_config == "std_grav_mag":
        sensor_cols = ["acc_mag_std", "grav_mag"]
    elif channel_config == "mag_std_grav_xz":
        sensor_cols = ["acc_mag", "acc_mag_std", "grav_x", "grav_z"]
    elif channel_config == "raw_mag_deriv_grav_xz":
        sensor_cols = acc_cols + ["acc_mag", "acc_dmag", "grav_x", "grav_z"]
    elif channel_config == "mag_std_grav_mag_xz":
        sensor_cols = ["acc_mag", "acc_mag_std", "grav_mag", "grav_x", "grav_z"]
    elif channel_config == "raw_grav":
        sensor_cols = acc_cols + ["grav_x", "grav_y", "grav_z"]
    elif channel_config == "raw_mag_deriv_grav":
        sensor_cols = acc_cols + ["acc_mag", "acc_dmag", "grav_x", "grav_y", "grav_z"]

    return df, sensor_cols


def build_label_mapping_fast(data_dir):
    labels = set()
    for path in list_client_files(data_dir):
        tqdm.write("Reading labels from " + os.path.basename(path) + "...")
        df = pd.read_csv(path, usecols=["activity"], dtype={"activity": "string"}, low_memory=False)
        labels.update(df["activity"].dropna().astype(str).unique().tolist())
    labels = sorted(labels)
    result = {}
    for idx, label in enumerate(labels):
        result[label] = idx
    return result


def build_dataset_label_sets(data_dir):
    dataset_labels = {}
    for path in list_client_files(data_dir):
        tqdm.write("Reading labels/dataset from " + os.path.basename(path) + "...")
        df = pd.read_csv(path, usecols=["activity", "dataset"], dtype={"activity": "string", "dataset": "string"}, low_memory=False)
        df = df.dropna(subset=["activity", "dataset"])
        for dataset_name, group in df.groupby("dataset"):
            labels = set(group["activity"].astype(str).unique().tolist())
            if dataset_name not in dataset_labels:
                dataset_labels[dataset_name] = labels
            else:
                dataset_labels[dataset_name].update(labels)
    return dataset_labels


def build_intersection_label_mapping(data_dir):
    csv_files = list_client_files(data_dir)
    probe = pd.read_csv(csv_files[0], nrows=1, low_memory=False)

    if "dataset" not in probe.columns:
        tqdm.write("WARNING: 'dataset' column not found. Falling back to union label mapping.")
        return build_label_mapping_fast(data_dir)

    dataset_label_sets = build_dataset_label_sets(data_dir)

    tqdm.write("\nFound " + str(len(dataset_label_sets)) + " datasets:")
    for name, lbls in sorted(dataset_label_sets.items()):
        tqdm.write("  " + name + ": " + str(sorted(lbls)))

    all_sets = list(dataset_label_sets.values())
    common_labels = all_sets[0].copy()
    for s in all_sets[1:]:
        common_labels &= s

    dropped = set()
    for s in all_sets:
        dropped.update(s - common_labels)

    tqdm.write("\nIntersection label mapping:")
    tqdm.write("  Kept (" + str(len(common_labels)) + "):    " + str(sorted(common_labels)))
    tqdm.write("  Dropped (" + str(len(dropped)) + "): " + str(sorted(dropped)))

    labels = sorted(common_labels)
    result = {}
    for idx, label in enumerate(labels):
        result[label] = idx
    return result


def estimate_sampling_rate_hz(df):
    ts = df["timestamp"]
    if not np.issubdtype(ts.dtype, np.number):
        ts = pd.to_datetime(ts)
        ts = ts.astype("int64") / 1e9
    dt_values = []
    for _, seg_df in df.groupby("segment_id", sort=False):
        seg_ts = pd.Series(ts.loc[seg_df.index]).reset_index(drop=True)
        dt = seg_ts.diff().dropna()
        dt = dt[dt > 0]
        if len(dt) > 0:
            dt_values.append(dt.median())
    median_dt = float(np.median(dt_values))
    if median_dt > 1.0:
        return 1000.0 / median_dt
    return 1.0 / median_dt


def detect_gravity_present(df, acc_cols=("acc_x", "acc_y", "acc_z")):
    present_cols = []
    for c in acc_cols:
        if c in df.columns:
            present_cols.append(c)
    if len(present_cols) == 0:
        return False
    means = []
    for c in present_cols:
        means.append(abs(df[c].mean()))
    for m in means:
        if m > 3.0:
            return True
    for m in means:
        if m > 0.3:
            return True
    return False


def estimate_gravity_lowpass(df, sampling_rate_hz, acc_cols=("acc_x", "acc_y", "acc_z"), cutoff_hz=0.3):
    df = df.copy()
    present_cols = []
    for c in acc_cols:
        if c in df.columns:
            present_cols.append(c)
    nyq = sampling_rate_hz / 2.0

    if cutoff_hz >= nyq:
        tqdm.write("  [gravity est.] cutoff >= Nyquist — gravity set to 0")
        for col in present_cols:
            axis = col[-1]
            df["grav_" + axis] = 0.0
        return df

    b, a = butter(N=3, Wn=cutoff_hz / nyq, btype="low")

    for col in present_cols:
        axis = col[-1]
        grav_col = "grav_" + axis
        df[grav_col] = 0.0
        for seg_id, seg_idx in df.groupby("segment_id", sort=False).groups.items():
            seg = df.loc[seg_idx]
            if len(seg) < 27:
                df.loc[seg_idx, grav_col] = seg[col].mean()
                continue
            df.loc[seg_idx, grav_col] = filtfilt(b, a, seg[col].values)

    return df


def remove_gravity_highpass(df, sampling_rate_hz, acc_cols=("acc_x", "acc_y", "acc_z"), cutoff_hz=0.3):
    df = df.copy()
    present_cols = []
    for c in acc_cols:
        if c in df.columns:
            present_cols.append(c)
    nyq = sampling_rate_hz / 2.0

    if cutoff_hz >= nyq:
        tqdm.write("  [gravity] cutoff >= Nyquist — skipping filter")
        return df

    b, a = butter(N=3, Wn=cutoff_hz / nyq, btype="high")

    for seg_id, seg_idx in df.groupby("segment_id", sort=False).groups.items():
        seg = df.loc[seg_idx]
        if len(seg) < 27:
            continue
        for col in present_cols:
            df.loc[seg_idx, col] = filtfilt(b, a, seg[col].values)

    return df


def standardize_acceleration_signal(df, dataset_name, sampling_rate_hz, datasets_in_g, datasets_gravity_removed, acc_cols=("acc_x", "acc_y", "acc_z")):
    df = df.copy()
    present_cols = []
    for c in acc_cols:
        if c in df.columns:
            present_cols.append(c)

    converted_units = dataset_name in datasets_in_g
    already_gravity_removed = dataset_name in datasets_gravity_removed
    gravity_removed_here = False

    if converted_units:
        for col in present_cols:
            df[col] = df[col] * 9.80665

    if not already_gravity_removed:
        df = estimate_gravity_lowpass(df, sampling_rate_hz=sampling_rate_hz, acc_cols=acc_cols, cutoff_hz=0.3)
        df = remove_gravity_highpass(df, sampling_rate_hz=sampling_rate_hz, acc_cols=acc_cols, cutoff_hz=0.3)
        gravity_removed_here = True
    else:
        for col in present_cols:
            axis = col[-1]
            df["grav_" + axis] = 0.0

    info = {
        "converted_units": converted_units,
        "already_gravity_removed": already_gravity_removed,
        "gravity_removed_here": gravity_removed_here,
        "acceleration_unit_after": "m/s^2",
        "acceleration_signal_after": "body_acceleration",
    }

    tqdm.write("  [" + dataset_name + "] converted_units=" + str(converted_units) + ", already_gravity_removed=" + str(already_gravity_removed) + ", gravity_removed_here=" + str(gravity_removed_here))

    return df, info


def train_val_split_by_segment(X, y, window_segment_ids, window_size, step_size, val_ratio=0.2, split_seed=42):
    gap = max(1, int(np.ceil(window_size / step_size)) - 1)

    X_train_parts = []
    X_val_parts = []
    y_train_parts = []
    y_val_parts = []
    seg_train_parts = []
    seg_val_parts = []

    for seg_id in pd.unique(window_segment_ids):
        mask = window_segment_ids == seg_id
        X_seg = X[mask]
        y_seg = y[mask]
        seg_ids = window_segment_ids[mask]
        n = len(X_seg)

        if n < 3:
            X_train_parts.append(X_seg)
            y_train_parts.append(y_seg)
            seg_train_parts.append(seg_ids)
            continue

        n_val = max(1, int(round(n * val_ratio)))
        n_train = n - n_val - gap

        if n_train <= 0:
            X_train_parts.append(X_seg)
            y_train_parts.append(y_seg)
            seg_train_parts.append(seg_ids)
            continue

        X_train_parts.append(X_seg[:n_train])
        y_train_parts.append(y_seg[:n_train])
        seg_train_parts.append(seg_ids[:n_train])

        X_val_parts.append(X_seg[n_train + gap:])
        y_val_parts.append(y_seg[n_train + gap:])
        seg_val_parts.append(seg_ids[n_train + gap:])

    if len(X_train_parts) > 0:
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        seg_train = np.concatenate(seg_train_parts, axis=0)
    else:
        X_train = np.empty((0,) + X.shape[1:])
        y_train = np.empty(0, dtype=y.dtype)
        seg_train = np.empty(0, dtype=window_segment_ids.dtype)

    if len(X_val_parts) > 0:
        X_val = np.concatenate(X_val_parts, axis=0)
        y_val = np.concatenate(y_val_parts, axis=0)
        seg_val = np.concatenate(seg_val_parts, axis=0)
    else:
        X_val = np.empty((0,) + X.shape[1:])
        y_val = np.empty(0, dtype=y.dtype)
        seg_val = np.empty(0, dtype=window_segment_ids.dtype)

    debug_info = {
        "train_classes_before": sorted(np.unique(y_train).tolist()) if len(y_train) > 0 else [],
        "val_classes_before": sorted(np.unique(y_val).tolist()) if len(y_val) > 0 else [],
        "rescued_classes": [],
        "val_rescued_classes": [],
        "cannot_rescue_to_val": [],
        "train_classes_after": [],
        "val_classes_after": [],
    }

    if len(y_train) > 0 and len(y_val) > 0:
        train_classes = set(np.unique(y_train).tolist())
        rescue_mask = np.array([c not in train_classes for c in y_val])
        if rescue_mask.any():
            rescued = sorted(np.unique(y_val[rescue_mask]).tolist())
            debug_info["rescued_classes"] = rescued
            X_train = np.concatenate([X_train, X_val[rescue_mask]], axis=0)
            y_train = np.concatenate([y_train, y_val[rescue_mask]], axis=0)
            seg_train = np.concatenate([seg_train, seg_val[rescue_mask]], axis=0)
            keep_mask = ~rescue_mask
            X_val = X_val[keep_mask]
            y_val = y_val[keep_mask]
            seg_val = seg_val[keep_mask]

    if len(y_train) > 0 and len(y_val) > 0:
        val_classes = set(np.unique(y_val).tolist())
        train_classes = set(np.unique(y_train).tolist())
        missing_from_val = sorted(train_classes - val_classes)

        val_rescued = []
        cannot_rescue = []

        for c in missing_from_val:
            train_idx_c = np.where(y_train == c)[0]
            if len(train_idx_c) < 2:
                cannot_rescue.append(c)
                continue
            move_idx = train_idx_c[-1]
            X_val = np.concatenate([X_val, X_train[move_idx:move_idx + 1]], axis=0)
            y_val = np.concatenate([y_val, y_train[move_idx:move_idx + 1]], axis=0)
            seg_val = np.concatenate([seg_val, seg_train[move_idx:move_idx + 1]], axis=0)
            keep_mask = np.ones(len(y_train), dtype=bool)
            keep_mask[move_idx] = False
            X_train = X_train[keep_mask]
            y_train = y_train[keep_mask]
            seg_train = seg_train[keep_mask]
            val_rescued.append(c)

        debug_info["val_rescued_classes"] = val_rescued
        debug_info["cannot_rescue_to_val"] = cannot_rescue

    debug_info["train_classes_after"] = sorted(np.unique(y_train).tolist()) if len(y_train) > 0 else []
    debug_info["val_classes_after"] = sorted(np.unique(y_val).tolist()) if len(y_val) > 0 else []

    return X_train, y_train, seg_train, X_val, y_val, seg_val, debug_info


def window_client_data(df, label_map, window_size, step_size, sensor_cols):
    X_parts = []
    y_parts = []
    seg_parts = []

    for seg_id, seg_df in df.groupby("segment_id", sort=False):
        seg_df = seg_df.reset_index(drop=True)
        n = len(seg_df)
        if n < window_size:
            continue

        data = seg_df[sensor_cols].values.astype(np.float32)
        labels_arr = seg_df["activity"].values

        windows = sliding_window_view(data, window_shape=(window_size, data.shape[1]))
        windows = windows[:, 0, :, :]
        windows = windows[::step_size]
        n_windows = windows.shape[0]

        label_windows = sliding_window_view(labels_arr, window_shape=window_size)[::step_size]

        for i in range(n_windows):
            win_labels = label_windows[i]
            unique = np.unique(win_labels)
            if len(unique) != 1:
                continue
            label_str = str(unique[0])
            if label_str not in label_map:
                continue
            X_parts.append(windows[i].T)
            y_parts.append(label_map[label_str])
            seg_parts.append(seg_id)

    if len(X_parts) == 0:
        return (
            np.empty((0, len(sensor_cols), window_size), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0),
        )

    return (
        np.stack(X_parts, axis=0).astype(np.float32),
        np.array(y_parts, dtype=np.int64),
        np.array(seg_parts),
    )


def apply_minmax_neg1_1(X, ch_min, ch_max):
    ch_range = ch_max - ch_min
    return np.where(
        ch_range > 0,
        2.0 * (X.astype(np.float64) - ch_min) / ch_range - 1.0,
        0.0,
    ).astype(np.float32)


def probe_client_eligible(csv_path, label_map):
    df = pd.read_csv(csv_path, low_memory=False)

    required = {"timestamp", "activity", "segment_id"}
    for col in required:
        if col not in df.columns:
            return None

    df = df.dropna(subset=["activity"])
    df = df[df["activity"].isin(label_map)]

    if len(df) == 0:
        return None

    if "dataset" in df.columns:
        return str(df["dataset"].iloc[0])
    return "unknown"


def parse_test_client_ids(s):
    if s is None or str(s).strip() == "":
        return None
    result = set()
    for x in str(s).split(","):
        if x.strip():
            result.add(int(x.strip()))
    return result


def parse_dataset_list(s):
    if s is None or str(s).strip() == "":
        return set()
    result = set()
    for x in str(s).split(","):
        if x.strip():
            result.add(x.strip())
    return result


def preprocess_train_client_unnorm(csv_path, label_map, window_seconds, overlap_ratio, channel_config, datasets_in_g, datasets_gravity_removed, val_ratio=0.2, split_seed=42):
    df = pd.read_csv(csv_path, low_memory=False)

    for col in ["timestamp", "activity", "segment_id"]:
        if col not in df.columns:
            tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": missing column " + col)
            return None

    df = df.dropna(subset=["activity"])
    df = df[df["activity"].isin(label_map)]

    if len(df) == 0:
        tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": no rows after label filtering")
        return None

    if "dataset" in df.columns:
        dataset_name = str(df["dataset"].iloc[0])
    else:
        dataset_name = "unknown"

    if "placement" in df.columns:
        placement = str(df["placement"].iloc[0])
    else:
        placement = "unknown"

    sampling_rate_hz = estimate_sampling_rate_hz(df)
    window_size = round(window_seconds * 20)
    step_size = max(1, round(window_size * (1.0 - overlap_ratio)))

    df, standardization_info = standardize_acceleration_signal(
        df=df,
        dataset_name=dataset_name,
        sampling_rate_hz=sampling_rate_hz,
        datasets_in_g=datasets_in_g,
        datasets_gravity_removed=datasets_gravity_removed,
    )

    df, sensor_cols = compute_derived_columns(df, channel_config, window_seconds)

    X, y, window_segment_ids = window_client_data(
        df=df,
        label_map=label_map,
        window_size=window_size,
        step_size=step_size,
        sensor_cols=sensor_cols,
    )

    if len(X) == 0:
        tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": no valid windows produced")
        return None

    X_train, y_train, seg_train, X_val, y_val, seg_val, debug_info = train_val_split_by_segment(
        X=X,
        y=y,
        window_segment_ids=window_segment_ids,
        window_size=window_size,
        step_size=step_size,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": train/val split produced empty set")
        return None

    train_class_counts = np.bincount(y_train, minlength=len(label_map)).tolist()
    val_class_counts = np.bincount(y_val, minlength=len(label_map)).tolist()

    tqdm.write("Finished " + os.path.basename(csv_path) + " — train: " + str(len(X_train)) + ", val: " + str(len(X_val)) + " (unnormalized)")

    return {
        "filename": os.path.basename(csv_path),
        "dataset": dataset_name,
        "placement": placement,
        "role": "train",
        "sensor_cols": sensor_cols,
        "n_rows": len(df),
        "n_sensor_cols": len(sensor_cols),
        "n_windows_total": len(X),
        "n_windows_train": len(X_train),
        "n_windows_val": len(X_val),
        "n_windows_test": 0,
        "n_classes_present": df["activity"].nunique(),
        "n_train_segments": len(pd.unique(seg_train)),
        "n_val_segments": len(pd.unique(seg_val)),
        "train_class_counts": train_class_counts,
        "val_class_counts": val_class_counts,
        "train_classes_before": debug_info["train_classes_before"],
        "val_classes_before": debug_info["val_classes_before"],
        "rescued_classes": debug_info["rescued_classes"],
        "val_rescued_classes": debug_info["val_rescued_classes"],
        "cannot_rescue_to_val": debug_info["cannot_rescue_to_val"],
        "train_classes_after": debug_info["train_classes_after"],
        "val_classes_after": debug_info["val_classes_after"],
        "sampling_rate_hz": sampling_rate_hz,
        "window_seconds": window_seconds,
        "overlap_ratio": overlap_ratio,
        "window_size_samples": window_size,
        "step_size_samples": step_size,
        "X_train": X_train,
        "y_train": y_train,
        "seg_train": seg_train,
        "X_val": X_val,
        "y_val": y_val,
        "seg_val": seg_val,
        "converted_units": standardization_info["converted_units"],
        "already_gravity_removed": standardization_info["already_gravity_removed"],
        "gravity_removed_here": standardization_info["gravity_removed_here"],
        "acceleration_unit_after": standardization_info["acceleration_unit_after"],
        "acceleration_signal_after": standardization_info["acceleration_signal_after"],
    }


def preprocess_test_client_unnorm(csv_path, label_map, window_seconds, overlap_ratio, channel_config, datasets_in_g, datasets_gravity_removed):
    df = pd.read_csv(csv_path, low_memory=False)

    for col in ["timestamp", "activity", "segment_id"]:
        if col not in df.columns:
            tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": missing column " + col)
            return None

    df = df.dropna(subset=["activity"])
    df = df[df["activity"].isin(label_map)]

    if len(df) == 0:
        tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": no rows after label filtering")
        return None

    if "dataset" in df.columns:
        dataset_name = str(df["dataset"].iloc[0])
    else:
        dataset_name = "unknown"

    if "placement" in df.columns:
        placement = str(df["placement"].iloc[0])
    else:
        placement = "unknown"

    sampling_rate_hz = estimate_sampling_rate_hz(df)
    window_size = round(window_seconds * 20)
    step_size = max(1, round(window_size * (1.0 - overlap_ratio)))

    df, standardization_info = standardize_acceleration_signal(
        df=df,
        dataset_name=dataset_name,
        sampling_rate_hz=sampling_rate_hz,
        datasets_in_g=datasets_in_g,
        datasets_gravity_removed=datasets_gravity_removed,
    )

    df, sensor_cols = compute_derived_columns(df, channel_config, window_seconds)

    X, y, window_segment_ids = window_client_data(
        df=df,
        label_map=label_map,
        window_size=window_size,
        step_size=step_size,
        sensor_cols=sensor_cols,
    )

    if len(X) == 0:
        tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": no valid windows produced")
        return None

    tqdm.write("Finished " + os.path.basename(csv_path) + " — test (all windows): " + str(len(X)) + " (unnormalized)")

    return {
        "filename": os.path.basename(csv_path),
        "dataset": dataset_name,
        "placement": placement,
        "role": "test",
        "sensor_cols": sensor_cols,
        "n_rows": len(df),
        "n_sensor_cols": len(sensor_cols),
        "n_windows_total": len(X),
        "n_windows_train": 0,
        "n_windows_val": 0,
        "n_windows_test": len(X),
        "n_classes_present": df["activity"].nunique(),
        "n_train_segments": 0,
        "n_val_segments": 0,
        "train_class_counts": [],
        "val_class_counts": [],
        "train_classes_before": [],
        "val_classes_before": [],
        "rescued_classes": [],
        "val_rescued_classes": [],
        "cannot_rescue_to_val": [],
        "train_classes_after": [],
        "val_classes_after": [],
        "sampling_rate_hz": sampling_rate_hz,
        "window_seconds": window_seconds,
        "overlap_ratio": overlap_ratio,
        "window_size_samples": window_size,
        "step_size_samples": step_size,
        "X_test": X,
        "y_test": y,
        "seg_test": window_segment_ids,
        "converted_units": standardization_info["converted_units"],
        "already_gravity_removed": standardization_info["already_gravity_removed"],
        "gravity_removed_here": standardization_info["gravity_removed_here"],
        "acceleration_unit_after": standardization_info["acceleration_unit_after"],
        "acceleration_signal_after": standardization_info["acceleration_signal_after"],
    }


def main(
    data_dir="data",
    output_dir="precomputed",
    window_seconds=5.0,
    overlap_ratio=0.5,
    val_ratio=0.2,
    split_seed=42,
    label_mapping="intersection",
    channel_config="mag_deriv_std",
    shuffle_windows=False,
    test_client_ids=None,
    inter_subject_test_ratio=0.2,
    inter_subject_split_seed=42,
    datasets_in_g=None,
    datasets_gravity_removed=None,
):
    datasets_in_g_set = parse_dataset_list(datasets_in_g)
    datasets_gravity_removed_set = parse_dataset_list(datasets_gravity_removed)

    tqdm.write("Datasets converted from g to m/s²: " + str(sorted(datasets_in_g_set)))
    tqdm.write("Datasets already gravity-removed: " + str(sorted(datasets_gravity_removed_set)))

    data_path = os.path.abspath(data_dir)
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)

    tqdm.write("Listing client files...")
    csv_files = list_client_files(data_path)

    if label_mapping == "intersection":
        tqdm.write("Building intersection label mapping...")
        label_map = build_intersection_label_mapping(data_path)
    else:
        tqdm.write("Building union label mapping...")
        label_map = build_label_mapping_fast(data_path)

    tqdm.write("Channel config: " + channel_config)
    tqdm.write("Shuffle windows: " + str(shuffle_windows))

    skipped_probe = []
    eligible = []
    for csv_path in csv_files:
        ds = probe_client_eligible(csv_path, label_map)
        if ds is None:
            skipped_probe.append(os.path.basename(csv_path))
            continue
        cid = len(eligible)
        eligible.append((cid, csv_path, ds))

    client_ids_list = []
    for e in eligible:
        client_ids_list.append(e[0])

    datasets_list = []
    for e in eligible:
        datasets_list.append(e[2])

    parsed_ids = parse_test_client_ids(test_client_ids)

    if parsed_ids != None:
        test_set = set(parsed_ids)
    elif inter_subject_test_ratio != None and inter_subject_test_ratio > 0:
        _, test_ids_list = inter_subject_split_client_ids(
            client_ids_list,
            datasets_list,
            inter_subject_test_ratio,
            inter_subject_split_seed,
        )
        test_set = set(test_ids_list)
    else:
        test_set = set()

    train_set = set(client_ids_list) - test_set

    tqdm.write("Client plan: " + str(len(train_set)) + " train-role, " + str(len(test_set)) + " test-role")

    staging = os.path.join(output_path, "_staging_unnorm")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)

    manifest_rows = []
    sensor_cols_global = None
    total_train_windows = 0
    total_val_windows = 0
    total_test_windows = 0
    total_all_windows = 0

    for cid, csv_path, dataset in tqdm(eligible, desc="Pass 1 — window & stage (unnorm)", unit="file"):
        is_test = cid in test_set
        if is_test:
            raw = preprocess_test_client_unnorm(
                csv_path=csv_path,
                label_map=label_map,
                window_seconds=window_seconds,
                overlap_ratio=overlap_ratio,
                channel_config=channel_config,
                datasets_in_g=datasets_in_g_set,
                datasets_gravity_removed=datasets_gravity_removed_set,
            )
        else:
            raw = preprocess_train_client_unnorm(
                csv_path=csv_path,
                label_map=label_map,
                window_seconds=window_seconds,
                overlap_ratio=overlap_ratio,
                channel_config=channel_config,
                val_ratio=val_ratio,
                split_seed=split_seed,
                datasets_in_g=datasets_in_g_set,
                datasets_gravity_removed=datasets_gravity_removed_set,
            )

        if raw is None:
            tqdm.write("SKIPPED " + os.path.basename(csv_path) + ": insufficient windows — dropping client.")
            continue

        if sensor_cols_global is None:
            sensor_cols_global = raw["sensor_cols"]

        if is_test:
            np.savez_compressed(
                os.path.join(staging, "client_" + str(cid).zfill(3) + "_test_raw.npz"),
                X=raw["X_test"],
                y=raw["y_test"],
                window_segment_ids=raw["seg_test"],
            )
            row = {"client_id": cid, "role": "test", "train_output_file": "", "val_output_file": "", "test_output_file": "client_" + str(cid).zfill(3) + "_test.npz"}
            for k, v in raw.items():
                if k not in ("X_test", "y_test", "seg_test", "role"):
                    row[k] = v
            manifest_rows.append(row)
            total_test_windows += raw["n_windows_test"]
        else:
            np.savez_compressed(
                os.path.join(staging, "client_" + str(cid).zfill(3) + "_train_raw.npz"),
                X=raw["X_train"],
                y=raw["y_train"],
                window_segment_ids=raw["seg_train"],
            )
            np.savez_compressed(
                os.path.join(staging, "client_" + str(cid).zfill(3) + "_val_raw.npz"),
                X=raw["X_val"],
                y=raw["y_val"],
                window_segment_ids=raw["seg_val"],
            )
            row = {"client_id": cid, "role": "train", "train_output_file": "client_" + str(cid).zfill(3) + "_train.npz", "val_output_file": "client_" + str(cid).zfill(3) + "_val.npz", "test_output_file": ""}
            for k, v in raw.items():
                if k not in ("X_train", "y_train", "seg_train", "X_val", "y_val", "seg_val", "role"):
                    row[k] = v
            manifest_rows.append(row)
            total_train_windows += raw["n_windows_train"]
            total_val_windows += raw["n_windows_val"]

        total_all_windows += raw["n_windows_total"]

    train_ids_sorted = []
    for r in manifest_rows:
        if r["role"] != "test":
            train_ids_sorted.append(r["client_id"])
    train_ids_sorted = sorted(train_ids_sorted)

    ch_min_g = None
    ch_max_g = None
    for cid in train_ids_sorted:
        path_t = os.path.join(staging, "client_" + str(cid).zfill(3) + "_train_raw.npz")
        Xtr = np.load(path_t, allow_pickle=True)["X"]
        tmin = Xtr.min(axis=(0, 2), keepdims=True)
        tmax = Xtr.max(axis=(0, 2), keepdims=True)
        if ch_min_g is None:
            ch_min_g = tmin
            ch_max_g = tmax
        else:
            ch_min_g = np.minimum(ch_min_g, tmin)
            ch_max_g = np.maximum(ch_max_g, tmax)

    norm_min_list = ch_min_g.squeeze().tolist()
    norm_max_list = ch_max_g.squeeze().tolist()

    tqdm.write("Pass 2 — apply global min-max [-1, 1] and write final .npz")
    for row in manifest_rows:
        cid = int(row["client_id"])
        if row["role"] == "test":
            z = np.load(os.path.join(staging, "client_" + str(cid).zfill(3) + "_test_raw.npz"), allow_pickle=True)
            Xn = apply_minmax_neg1_1(z["X"], ch_min_g, ch_max_g)
            np.savez_compressed(
                os.path.join(output_path, "client_" + str(cid).zfill(3) + "_test.npz"),
                X=Xn,
                y=z["y"],
                window_segment_ids=z["window_segment_ids"],
            )
        else:
            z_tr = np.load(os.path.join(staging, "client_" + str(cid).zfill(3) + "_train_raw.npz"), allow_pickle=True)
            z_va = np.load(os.path.join(staging, "client_" + str(cid).zfill(3) + "_val_raw.npz"), allow_pickle=True)
            X_train_n = apply_minmax_neg1_1(z_tr["X"], ch_min_g, ch_max_g)
            X_val_n = apply_minmax_neg1_1(z_va["X"], ch_min_g, ch_max_g)
            y_train = z_tr["y"]
            seg_train = z_tr["window_segment_ids"]
            if shuffle_windows:
                rng = np.random.default_rng(split_seed)
                shuffle_idx = rng.permutation(len(X_train_n))
                X_train_n = X_train_n[shuffle_idx]
                y_train = y_train[shuffle_idx]
                seg_train = seg_train[shuffle_idx]
            np.savez_compressed(
                os.path.join(output_path, "client_" + str(cid).zfill(3) + "_train.npz"),
                X=X_train_n,
                y=y_train,
                window_segment_ids=seg_train,
            )
            np.savez_compressed(
                os.path.join(output_path, "client_" + str(cid).zfill(3) + "_val.npz"),
                X=X_val_n,
                y=z_va["y"],
                window_segment_ids=z_va["window_segment_ids"],
            )

    shutil.rmtree(staging, ignore_errors=True)

    for row in manifest_rows:
        row["norm_min"] = norm_min_list
        row["norm_max"] = norm_max_list
        row["norm_target_range"] = [-1.0, 1.0]

    if len(skipped_probe) > 0:
        tqdm.write("\n" + str(len(skipped_probe)) + " client file(s) skipped at probe:")
        for name in skipped_probe:
            tqdm.write("  - " + name)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(output_path, "manifest.csv"), index=False)

    all_labels_union = set(build_label_mapping_fast(data_path).keys())
    dropped_labels = sorted(all_labels_union - set(label_map.keys()))
    n_kept = len(manifest_rows)

    save_json(
        {
            "data_dir": data_path,
            "output_dir": output_path,
            "window_seconds": window_seconds,
            "overlap_ratio": overlap_ratio,
            "val_ratio": val_ratio,
            "split_seed": split_seed,
            "label_mapping": label_mapping,
            "channel_config": channel_config,
            "shuffle_windows": shuffle_windows,
            "inter_subject_test_ratio": inter_subject_test_ratio,
            "inter_subject_split_seed": inter_subject_split_seed,
            "train_client_ids": sorted(train_set),
            "test_client_ids": sorted(test_set),
            "num_clients_total": len(csv_files),
            "num_clients_kept": n_kept,
            "num_clients_skipped_probe": len(skipped_probe),
            "skipped_probe_files": skipped_probe,
            "label_map": label_map,
            "sensor_cols": sensor_cols_global,
            "window_label_policy": "pure_only",
            "windowing_policy": "time_based_per_file_sampling_rate",
            "kept_labels": sorted(label_map.keys()),
            "dropped_labels": dropped_labels,
            "split_policy": "inter_subject_train_clients_segment_level_val_test_clients_all_windows",
            "global_norm_min": norm_min_list,
            "global_norm_max": norm_max_list,
            "normalization": {
                "kind": "channel_wise_minmax",
                "fit_on": "all_train_role_clients_train_splits_combined",
                "target_range": [-1.0, 1.0],
                "constant_channel_output": 0.0,
            },
            "total_windows": total_all_windows,
            "total_train_windows": total_train_windows,
            "total_val_windows": total_val_windows,
            "total_test_windows": total_test_windows,
            "datasets_in_g": sorted(datasets_in_g_set),
            "datasets_gravity_removed": sorted(datasets_gravity_removed_set),
            "acceleration_standardization": {
                "unit_after": "m/s^2",
                "signal_after": "body_acceleration",
                "g_to_ms2_factor": 9.80665,
                "gravity_removal_method": "highpass_butterworth",
                "gravity_estimation_method": "lowpass_butterworth",
                "gravity_cutoff_hz": 0.3,
            },
        },
        os.path.join(output_path, "metadata.json"),
    )

    tqdm.write("\nSaved manifest to: " + os.path.join(output_path, "manifest.csv"))
    tqdm.write("Saved metadata to: " + os.path.join(output_path, "metadata.json"))
    tqdm.write("Clients processed: " + str(n_kept) + " kept, " + str(len(skipped_probe)) + " skipped at probe")
    tqdm.write("Total train / val / test windows: " + str(total_train_windows) + " / " + str(total_val_windows) + " / " + str(total_test_windows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="precomputed")
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--label-mapping", type=str, default="intersection")
    parser.add_argument("--channel-config", type=str, default="mag_std")
    parser.add_argument("--shuffle-windows", action="store_true", default=False)
    parser.add_argument("--test-client-ids", type=str, default=None)
    parser.add_argument("--inter-subject-test-ratio", type=float, default=None)
    parser.add_argument("--inter-subject-split-seed", type=int, default=42)
    parser.add_argument("--datasets-in-g", type=str, default="HAPT")
    parser.add_argument("--datasets-gravity-removed", type=str, default="")
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        window_seconds=args.window_seconds,
        overlap_ratio=args.overlap_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        label_mapping=args.label_mapping,
        channel_config=args.channel_config,
        shuffle_windows=args.shuffle_windows,
        test_client_ids=args.test_client_ids,
        inter_subject_test_ratio=args.inter_subject_test_ratio,
        inter_subject_split_seed=args.inter_subject_split_seed,
        datasets_in_g=args.datasets_in_g,
        datasets_gravity_removed=args.datasets_gravity_removed,
    )