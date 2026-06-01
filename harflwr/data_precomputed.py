import json
import math
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# tested different channel configs during exploration
# but only mag_std was used for final experiments after analysis
CHANNEL_CONFIGS = (
    "raw", "raw_mag", "raw_mag_deriv", "mag", "deriv", "mag_deriv",
    "raw_deriv", "mag_std", "mag_deriv_std", "std",
    "raw_grav", "raw_mag_deriv_grav", "mag_std_grav", "std_grav",
    "mag_std_grav_mag", "std_grav_mag",
    "mag_std_grav_xz", "std_grav_xz", "raw_mag_deriv_grav_xz", "mag_std_grav_mag_xz",
)


def get_precomputed_dir(precomputed_dir=None):
    if precomputed_dir is None or str(precomputed_dir).strip() == "":
        return os.path.join(os.getcwd(), "precomputed")
    return os.path.abspath(precomputed_dir)


def read_metadata(precomputed_dir=None):
    pre_path = get_precomputed_dir(precomputed_dir)
    metadata_path = os.path.join(pre_path, "metadata.json")
    f = open(metadata_path, "r")
    data = json.load(f)
    f.close()
    return data


def read_manifest(precomputed_dir=None):
    pre_path = get_precomputed_dir(precomputed_dir)
    manifest_path = os.path.join(pre_path, "manifest.csv")
    return pd.read_csv(manifest_path)


def invert_label_mapping(label_map):
    result = {}
    for k, v in label_map.items():
        result[v] = k
    return result


def extract_client_id_from_path(path):
    filename = os.path.basename(path)
    stem = filename.replace(".npz", "")
    parts = stem.split("_")
    return int(parts[1])


def get_train_test_client_ids(precomputed_dir=None, test_ratio=0.2, seed=42):
    pre_path = get_precomputed_dir(precomputed_dir)
    meta_path = os.path.join(pre_path, "metadata.json")

    if os.path.exists(meta_path):
        f = open(meta_path, "r")
        meta = json.load(f)
        f.close()

        tr = meta.get("train_client_ids")
        te = meta.get("test_client_ids")

        if tr != None and te != None:
            all_files = os.listdir(pre_path)

            existing_train = []
            for f in all_files:
                if f.startswith("client_") and f.endswith("_train.npz"):
                    path = os.path.join(pre_path, f)
                    existing_train.append(extract_client_id_from_path(path))

            existing_test = []
            for f in all_files:
                if f.startswith("client_") and f.endswith("_test.npz"):
                    path = os.path.join(pre_path, f)
                    existing_test.append(extract_client_id_from_path(path))

            tr_filtered = []
            for x in tr:
                if int(x) in existing_train:
                    tr_filtered.append(int(x))

            te_filtered = []
            for x in te:
                if int(x) in existing_test:
                    te_filtered.append(int(x))

            return sorted(tr_filtered), sorted(te_filtered)

    manifest = read_manifest(precomputed_dir)
    rng = np.random.default_rng(seed)
    train_client_ids = []
    test_client_ids = []

    for dataset_name, group in manifest.groupby("dataset"):
        ids = group["client_id"].tolist()
        shuffled = rng.permutation(ids)

        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        n_test = min(n_test, len(shuffled) - 1)

        for x in shuffled[-n_test:]:
            test_client_ids.append(int(x))
        for x in shuffled[:-n_test]:
            train_client_ids.append(int(x))

    return sorted(train_client_ids), sorted(test_client_ids)


def inter_subject_split_client_ids(client_ids, dataset_per_client, test_ratio, seed):
    df = pd.DataFrame({"client_id": client_ids, "dataset": dataset_per_client})
    rng = np.random.default_rng(seed)
    train_client_ids = []
    test_client_ids = []

    for dataset_name, group in df.groupby("dataset"):
        ids = group["client_id"].tolist()
        shuffled = rng.permutation(ids)

        n_test = max(1, int(round(len(shuffled) * test_ratio)))
        n_test = min(n_test, len(shuffled) - 1)

        for x in shuffled[-n_test:]:
            test_client_ids.append(int(x))
        for x in shuffled[:-n_test]:
            train_client_ids.append(int(x))

    return sorted(train_client_ids), sorted(test_client_ids)


def get_client_filepaths_from_subset(subset_client_ids, subset_partition_id, precomputed_dir=None):
    actual_client_id = subset_client_ids[subset_partition_id]
    pre_path = get_precomputed_dir(precomputed_dir)
    train_path = os.path.join(pre_path, "client_" + str(actual_client_id).zfill(3) + "_train.npz")
    val_path = os.path.join(pre_path, "client_" + str(actual_client_id).zfill(3) + "_val.npz")
    return train_path, val_path


def get_client_test_filepath(client_id, precomputed_dir=None):
    pre_path = get_precomputed_dir(precomputed_dir)
    test_path = os.path.join(pre_path, "client_" + str(client_id).zfill(3) + "_test.npz")
    if os.path.exists(test_path):
        return test_path
    all_path = os.path.join(pre_path, "client_" + str(client_id).zfill(3) + "_all.npz")
    return all_path


def select_channels(X, sensor_cols, channel_config):
    raw_cols = []
    for c in sensor_cols:
        if c in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
            raw_cols.append(c)

    mag_cols = []
    for c in sensor_cols:
        if c == "acc_mag":
            mag_cols.append(c)

    deriv_cols = []
    for c in sensor_cols:
        if c == "acc_dmag":
            deriv_cols.append(c)

    std_cols = []
    for c in sensor_cols:
        if c == "acc_mag_std":
            std_cols.append(c)

    grav_cols = []
    for c in sensor_cols:
        if c in ("grav_x", "grav_y", "grav_z"):
            grav_cols.append(c)

    grav_cols_xz = []
    for c in sensor_cols:
        if c in ("grav_x", "grav_z"):
            grav_cols_xz.append(c)

    grav_mag_col = []
    for c in sensor_cols:
        if c == "grav_mag":
            grav_mag_col.append(c)

    if channel_config == "raw":
        selected = raw_cols
    elif channel_config == "raw_mag":
        selected = raw_cols + mag_cols
    elif channel_config == "raw_mag_deriv":
        selected = raw_cols + mag_cols + deriv_cols
    elif channel_config == "mag":
        selected = mag_cols
    elif channel_config == "deriv":
        selected = deriv_cols
    elif channel_config == "std":
        selected = std_cols
    elif channel_config == "mag_deriv":
        selected = mag_cols + deriv_cols
    elif channel_config == "raw_deriv":
        selected = raw_cols + deriv_cols
    elif channel_config == "mag_std":
        selected = mag_cols + std_cols
    elif channel_config == "mag_deriv_std":
        selected = mag_cols + deriv_cols + std_cols
    elif channel_config == "raw_grav":
        selected = raw_cols + grav_cols
    elif channel_config == "raw_mag_deriv_grav":
        selected = raw_cols + mag_cols + deriv_cols + grav_cols
    elif channel_config == "mag_std_grav":
        selected = mag_cols + std_cols + grav_cols
    elif channel_config == "std_grav":
        selected = std_cols + grav_cols
    elif channel_config == "mag_std_grav_mag":
        selected = mag_cols + std_cols + grav_mag_col
    elif channel_config == "std_grav_mag":
        selected = std_cols + grav_mag_col
    elif channel_config == "mag_std_grav_xz":
        selected = mag_cols + std_cols + grav_cols_xz
    elif channel_config == "std_grav_xz":
        selected = std_cols + grav_cols_xz
    elif channel_config == "raw_mag_deriv_grav_xz":
        selected = raw_cols + mag_cols + deriv_cols + grav_cols_xz
    elif channel_config == "mag_std_grav_mag_xz":
        selected = mag_cols + std_cols + grav_mag_col + grav_cols_xz

    idx = []
    for c in selected:
        idx.append(sensor_cols.index(c))

    return X[:, idx, :], selected


def make_loader(X, y, batch_size, shuffle, seed=42, drop_last=False):
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ds = TensorDataset(x_tensor, y_tensor)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=drop_last)


def load_client_data(
    partition_id,
    window_seconds,
    overlap_ratio,
    batch_size=32,
    val_ratio=0.2,
    precomputed_dir=None,
    selected_client_ids=None,
    server_round=None,
    chunk_size=32,
    channel_config="raw_mag_deriv",
):
    if selected_client_ids is None:
        pre_path = get_precomputed_dir(precomputed_dir)
        train_path = os.path.join(pre_path, "client_" + str(partition_id).zfill(3) + "_train.npz")
        val_path = os.path.join(pre_path, "client_" + str(partition_id).zfill(3) + "_val.npz")
    else:
        train_path, val_path = get_client_filepaths_from_subset(
            subset_client_ids=selected_client_ids,
            subset_partition_id=partition_id,
            precomputed_dir=precomputed_dir,
        )

    train_arr = np.load(train_path, allow_pickle=True)
    val_arr = np.load(val_path, allow_pickle=True)

    X_train = train_arr["X"]
    y_train = train_arr["y"]
    X_val = val_arr["X"]
    y_val = val_arr["y"]

    metadata = read_metadata(precomputed_dir)
    sensor_cols = metadata["sensor_cols"]
    label_map = metadata["label_map"]

    stored_config = metadata.get("channel_config", "")
    if channel_config and channel_config != stored_config:
        original_sensor_cols = sensor_cols
        X_train, sensor_cols = select_channels(X_train, original_sensor_cols, channel_config)
        X_val, _ = select_channels(X_val, original_sensor_cols, channel_config)

    if server_round != None:
        start = (server_round - 1) * chunk_size
        end = start + chunk_size
        X_train = X_train[start:end]
        y_train = y_train[start:end]

    present_classes = np.unique(y_train).tolist()

    trainloader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False, drop_last=False)
    num_channels = X_train.shape[1]
    num_classes = len(label_map)

    return trainloader, valloader, num_channels, num_classes, sensor_cols, label_map, present_classes


def load_client_test_data(
    partition_id,
    window_seconds=5.0,
    overlap_ratio=0.5,
    batch_size=32,
    precomputed_dir=None,
    selected_client_ids=None,
    channel_config="raw_mag_deriv",
):
    client_id = partition_id
    test_path = get_client_test_filepath(client_id, precomputed_dir=precomputed_dir)

    arr = np.load(test_path, allow_pickle=True)
    X = arr["X"]
    y = arr["y"]

    metadata = read_metadata(precomputed_dir)
    sensor_cols = metadata["sensor_cols"]
    label_map = metadata["label_map"]

    stored_config = metadata.get("channel_config", "")
    if channel_config and channel_config != stored_config:
        X, sensor_cols = select_channels(X, sensor_cols, channel_config)

    present_classes = np.unique(y).tolist()
    testloader = make_loader(X, y, batch_size=batch_size, shuffle=False)
    num_channels = X.shape[1]
    num_classes = len(label_map)

    return testloader, num_channels, num_classes, sensor_cols, label_map, present_classes


def load_all_data_centralized_arrays(precomputed_dir=None, selected_client_ids=None):
    X_train_all = []
    y_train_all = []

    metadata = read_metadata(precomputed_dir)
    label_map = metadata["label_map"]
    sensor_cols_global = metadata["sensor_cols"]

    if selected_client_ids is None:
        pre_path = get_precomputed_dir(precomputed_dir)
        all_files = os.listdir(pre_path)
        client_ids = []
        for f in all_files:
            if f.startswith("client_") and f.endswith("_train.npz"):
                path = os.path.join(pre_path, f)
                client_ids.append(extract_client_id_from_path(path))
        client_ids = sorted(client_ids)
    else:
        client_ids = selected_client_ids

    for i in range(len(client_ids)):
        train_path, val_path = get_client_filepaths_from_subset(
            subset_client_ids=client_ids,
            subset_partition_id=i,
            precomputed_dir=precomputed_dir,
        )

        train_arr = np.load(train_path, allow_pickle=True)
        X_train = train_arr["X"]
        y_train = train_arr["y"]

        if len(X_train) == 0:
            continue

        X_train_all.append(X_train)
        y_train_all.append(y_train)

    X_all = np.concatenate(X_train_all, axis=0)
    y_all = np.concatenate(y_train_all, axis=0)

    return X_all, y_all, sensor_cols_global, label_map


def compute_client_total_chunks(train_client_ids, precomputed_dir, chunk_size):
    pre_path = get_precomputed_dir(precomputed_dir)

    total_chunks = {}
    for cid in train_client_ids:
        train_path = os.path.join(pre_path, "client_" + str(cid).zfill(3) + "_train.npz")
        arr = np.load(train_path, allow_pickle=True)
        n_windows = len(arr["X"])
        total_chunks[cid] = max(1, math.ceil(n_windows / chunk_size))

    return total_chunks