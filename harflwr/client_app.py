import json
import time
import shutil
import os

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader, TensorDataset

from harflwr.data_precomputed import load_client_data
from harflwr.task import evaluate_fn, get_model, train_fn

app = ClientApp()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_train_client_ids(config_value):
    result = []
    for x in config_value.split(","):
        if x.strip():
            result.append(int(x))
    return result


def summarize_state_dict(state_dict):
    total_sq = 0.0
    for t in state_dict.values():
        norm = torch.norm(t.detach().float().cpu()).item()
        total_sq += norm ** 2
    return {"global_model_total_l2": total_sq ** 0.5}


def compute_per_class_val_stats(model, dataloader, num_classes, y_true_all, y_pred_all):
    total_counts = {}
    correct_counts = {}
    recall_per_class = {}

    for c in range(num_classes):
        mask = y_true_all == c
        total = int(mask.sum())
        correct = int(((y_pred_all == c) & mask).sum())
        total_counts[str(c)] = total
        correct_counts[str(c)] = correct
        if total > 0:
            recall_per_class[str(c)] = correct / total
        else:
            recall_per_class[str(c)] = float("nan")

    return {
        "val_class_total_counts": total_counts,
        "val_class_correct_counts": correct_counts,
        "val_class_recall_per_class": recall_per_class,
    }


def make_loader_from_arrays(X, y, batch_size, shuffle, seed=42, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=drop_last)


def is_personal_param(name):
    return name.startswith("classifier.")


def split_shared_state(state_dict):
    result = {}
    for k, v in state_dict.items():
        if not is_personal_param(k):
            result[k] = v
    return result


def split_personal_state(state_dict):
    result = {}
    for k, v in state_dict.items():
        if is_personal_param(k):
            result[k] = v
    return result


def get_personal_head_path(precomputed_dir, client_id):
    d = os.path.join(precomputed_dir, "_personal_heads")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "client_" + str(client_id).zfill(3) + "_head.pt")


def get_replay_path(precomputed_dir, client_id):
    replay_dir = os.path.join(precomputed_dir, "_replay_buffers")
    os.makedirs(replay_dir, exist_ok=True)
    return os.path.join(replay_dir, "client_" + str(client_id).zfill(3) + "_replay.npz")


def load_replay_buffer(precomputed_dir, client_id, num_classes):
    path = get_replay_path(precomputed_dir, client_id)
    if not os.path.exists(path):
        return {}
    arr = np.load(path, allow_pickle=True)
    replay_dict = {}
    for c in range(num_classes):
        x_key = "X_" + str(c)
        y_key = "y_" + str(c)
        if x_key in arr and y_key in arr and len(arr[x_key]) > 0:
            replay_dict[c] = (arr[x_key], arr[y_key])
    return replay_dict


def save_replay_buffer(precomputed_dir, client_id, replay_dict):
    path = get_replay_path(precomputed_dir, client_id)
    tmp_path = path.replace(".npz", ".tmp.npz")

    arrays = {}
    for cls_idx, (X, y) in replay_dict.items():
        arrays["X_" + str(cls_idx)] = X
        arrays["y_" + str(cls_idx)] = y
    np.savez_compressed(tmp_path, **arrays)

    for attempt in range(5):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))

    shutil.copy2(tmp_path, path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def update_replay_buffer(X_current, y_current, replay_dict, capacity_per_class, num_classes):
    new_dict = {}
    for k, v in replay_dict.items():
        new_dict[k] = v

    for c in range(num_classes):
        idx = np.where(y_current == c)[0]
        if len(idx) == 0:
            continue
        Xc = X_current[idx]
        yc = y_current[idx]
        if len(Xc) > capacity_per_class:
            Xc = Xc[-capacity_per_class:]
            yc = yc[-capacity_per_class:]
        new_dict[c] = (Xc.copy(), yc.copy())
    return new_dict


def sample_replay_class_aware(replay_dict, sample_size, current_chunk_class_ids, seed):
    if not replay_dict or sample_size <= 0:
        return None, None, {"replay_missing_class_windows_used": 0, "replay_class_counts_used": {}}

    rng = np.random.default_rng(seed)

    current_class_set = set()
    for c in current_chunk_class_ids:
        current_class_set.add(int(c))

    missing_classes = []
    for c in sorted(replay_dict.keys()):
        if c not in current_class_set:
            missing_classes.append(c)

    if len(missing_classes) == 0:
        return None, None, {"replay_missing_class_windows_used": 0, "replay_class_counts_used": {}}

    base_quota = sample_size // len(missing_classes)
    X_parts = []
    y_parts = []
    total_used = 0
    per_class_used = {}
    leftovers = []

    for c in missing_classes:
        Xc, yc = replay_dict[c]
        n_available = len(Xc)
        n_take = min(base_quota, n_available)

        if n_take > 0:
            chosen_idx = rng.choice(n_available, size=n_take, replace=False)
            X_parts.append(Xc[chosen_idx])
            y_parts.append(yc[chosen_idx])
            total_used += n_take
            if c in per_class_used:
                per_class_used[c] = per_class_used[c] + n_take
            else:
                per_class_used[c] = n_take
            remaining_idx = np.setdiff1d(np.arange(n_available), chosen_idx)
        else:
            remaining_idx = np.arange(n_available)

        if len(remaining_idx) > 0:
            leftovers.append((c, remaining_idx))

    extra_needed = sample_size - total_used
    while extra_needed > 0 and len(leftovers) > 0:
        new_leftovers = []
        for c, remaining_idx in leftovers:
            if extra_needed <= 0:
                new_leftovers.append((c, remaining_idx))
                continue
            if len(remaining_idx) == 0:
                continue
            chosen_one = rng.choice(remaining_idx, size=1, replace=False)
            Xc, yc = replay_dict[c]
            X_parts.append(Xc[chosen_one])
            y_parts.append(yc[chosen_one])
            total_used += 1
            extra_needed -= 1
            if c in per_class_used:
                per_class_used[c] = per_class_used[c] + 1
            else:
                per_class_used[c] = 1
            still_left = np.setdiff1d(remaining_idx, chosen_one)
            if len(still_left) > 0:
                new_leftovers.append((c, still_left))
        leftovers = new_leftovers

    if len(X_parts) == 0:
        return None, None, {"replay_missing_class_windows_used": 0, "replay_class_counts_used": {}}

    X_sample = np.concatenate(X_parts, axis=0)
    y_sample = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(X_sample))

    per_class_used_str = {}
    for k, v in per_class_used.items():
        per_class_used_str[str(k)] = v

    return (
        X_sample[perm],
        y_sample[perm],
        {
            "replay_missing_class_windows_used": total_used,
            "replay_class_counts_used": per_class_used_str,
        },
    )


def exhausted_reply(msg, client_id, server_round, num_classes):
    metrics = MetricRecord({
        "client_id": client_id,
        "server_round": server_round,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_accuracy": 0.0,
        "val_macro_f1": 0.0,
        "val_weighted_f1": 0.0,
        "num-examples": 0,
        "fit_duration_sec": 0.0,
        "local_eval_duration_sec": 0.0,
        "chunk_start_idx": -1.0,
        "chunk_end_idx": -1.0,
        "chunk_class_counts": [0.0] * num_classes,
        "chunk_present_class_ids": [],
        "val_present_class_ids": [],
    })
    str_record = ConfigRecord({
        "replay_class_counts_used":         "{}",
        "val_class_recall_per_class_named": "{}",
    })
    return Message(
        content=RecordDict({
            "arrays": msg.content["arrays"],
            "metrics": metrics,
            "str_metrics": str_record,
        }),
        reply_to=msg,
    )


@app.train()
def train(msg, context):
    partition_id = context.node_config["partition-id"]

    window_seconds = float(context.run_config["window-seconds"])
    overlap_ratio = float(context.run_config["overlap-ratio"])
    batch_size = int(context.run_config["batch-size"])
    precomputed_dir = str(context.run_config["precomputed-dir"])
    chunk_size = int(context.run_config["chunk-size"])
    local_epochs = int(msg.content["config"]["local-epochs"])
    learning_rate = float(msg.content["config"]["learning-rate"])
    server_round = msg.content["config"]["server-round"]
    model_name = msg.content["config"]["model_name"]
    train_client_ids = parse_train_client_ids(msg.content["config"]["train-client-ids"])
    actual_client_id = int(train_client_ids[partition_id])
    num_classes = int(msg.content["config"]["num_classes"])
    label_map = json.loads(msg.content["config"]["label_map"])
    channel_config = str(context.run_config["channel-config"])

    inv_label_map = {}
    for k, v in label_map.items():
        inv_label_map[int(v)] = k

    use_replay_buffer = bool(int(msg.content["config"]["use_replay_buffer"]))
    replay_buffer_capacity = int(msg.content["config"]["replay_buffer_capacity"])
    replay_sample_size = int(msg.content["config"]["replay_sample_size"])

    selection_counts = json.loads(msg.content["config"]["client-selection-counts"])
    client_selection_round = selection_counts.get(str(actual_client_id), 0) + 1

    total_chunks_map = json.loads(msg.content["config"]["client-total-chunks"])
    total_chunks = total_chunks_map.get(str(actual_client_id), 1)

    personalization_mode = str(msg.content["config"]["personalization_mode"])

    window_size = int(window_seconds * 20)

    if client_selection_round > total_chunks:
        return exhausted_reply(msg, actual_client_id, server_round, num_classes)

    trainloader, valloader, num_channels, num_classes, _, _, _ = load_client_data(
        partition_id=partition_id,
        window_seconds=window_seconds,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
        precomputed_dir=precomputed_dir,
        selected_client_ids=train_client_ids,
        server_round=client_selection_round,
        chunk_size=chunk_size,
        channel_config=channel_config,
    )
    current_X = trainloader.dataset.tensors[0].cpu().numpy()
    current_y = trainloader.dataset.tensors[1].cpu().numpy()

    chunk_class_counts_list = np.bincount(current_y, minlength=num_classes).tolist()
    chunk_present_class_ids = []
    for i, c in enumerate(chunk_class_counts_list):
        if c > 0:
            chunk_present_class_ids.append(i)

    if sum(chunk_class_counts_list) > 0:
        chunk_majority_class_id = int(np.argmax(chunk_class_counts_list))
    else:
        chunk_majority_class_id = -1

    replay_windows_used = 0
    replay_missing_class_windows_used = 0
    replay_class_counts_used = {}

    if use_replay_buffer:
        replay_dict = load_replay_buffer(precomputed_dir, actual_client_id, num_classes)

        X_replay_sample, y_replay_sample, replay_stats = sample_replay_class_aware(
            replay_dict=replay_dict,
            sample_size=replay_sample_size,
            current_chunk_class_ids=chunk_present_class_ids,
            seed=server_round + actual_client_id,
        )

        replay_missing_class_windows_used = replay_stats["replay_missing_class_windows_used"]
        replay_class_counts_used = replay_stats["replay_class_counts_used"]

        if X_replay_sample is not None:
            replay_windows_used = len(X_replay_sample)
            X_train_mix = np.concatenate([current_X, X_replay_sample], axis=0)
            y_train_mix = np.concatenate([current_y, y_replay_sample], axis=0)
        else:
            X_train_mix = current_X
            y_train_mix = current_y

        trainloader = make_loader_from_arrays(
            X=X_train_mix, y=y_train_mix,
            batch_size=batch_size, shuffle=True,
            seed=server_round + actual_client_id,
            drop_last=True,
        )
    else:
        replay_dict = {}
        X_train_mix = current_X
        y_train_mix = current_y

    chunk_start_idx = (client_selection_round - 1) * chunk_size
    chunk_end_idx = chunk_start_idx + len(current_y)

    model = get_model(model_name, num_channels, num_classes, window_size)
    state = msg.content["arrays"].to_torch_state_dict()

    global_model_summary = summarize_state_dict(state)

    if personalization_mode == "fedper":
        model.load_state_dict(state, strict=False)
        head_path = get_personal_head_path(precomputed_dir, actual_client_id)
        if os.path.exists(head_path):
            personal_state = torch.load(head_path, map_location=DEVICE)
            model_state = model.state_dict()
            model_state.update(personal_state)
            model.load_state_dict(model_state, strict=True)
    else:
        model.load_state_dict(state, strict=True)

    model.to(DEVICE)

    fedper_adapt_epochs = int(msg.content["config"]["fedper_adapt_epochs"])

    t0 = time.perf_counter()

    if personalization_mode == "fedper":
        for name, param in model.named_parameters():
            param.requires_grad = is_personal_param(name)

        train_fn(
            model=model,
            trainloader=trainloader,
            epochs=fedper_adapt_epochs,
            device=DEVICE,
            learning_rate=learning_rate,
        )

        for param in model.parameters():
            param.requires_grad = True

    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=local_epochs,
        device=DEVICE,
        learning_rate=learning_rate,
    )
    fit_duration = time.perf_counter() - t0

    if use_replay_buffer:
        replay_dict = update_replay_buffer(
            X_current=current_X,
            y_current=current_y,
            replay_dict=replay_dict,
            capacity_per_class=replay_buffer_capacity,
            num_classes=num_classes,
        )
        save_replay_buffer(precomputed_dir, actual_client_id, replay_dict)
        replay_buffer_size_after_update = 0
        for _, yc in replay_dict.values():
            replay_buffer_size_after_update += len(yc)
    else:
        replay_buffer_size_after_update = 0

    if personalization_mode == "fedper":
        personal_state = split_personal_state(model.state_dict())
        save_dict = {}
        for k, v in personal_state.items():
            save_dict[k] = v.detach().cpu()
        torch.save(save_dict, get_personal_head_path(precomputed_dir, actual_client_id))

    eval_t0 = time.perf_counter()
    eval_metrics = evaluate_fn(model=model, dataloader=valloader, device=DEVICE)
    local_eval_duration = time.perf_counter() - eval_t0

    y_true_val = eval_metrics["y_true"]
    y_pred_val = eval_metrics["y_pred"]

    per_class_stats = compute_per_class_val_stats(
        model=model,
        dataloader=valloader,
        num_classes=num_classes,
        y_true_all=y_true_val,
        y_pred_all=y_pred_val,
    )

    val_class_counts_list = np.bincount(y_true_val, minlength=num_classes).tolist()
    val_present_class_ids = []
    for i, c in enumerate(val_class_counts_list):
        if c > 0:
            val_present_class_ids.append(i)

    if sum(val_class_counts_list) > 0:
        val_majority_class_id = int(np.argmax(val_class_counts_list))
    else:
        val_majority_class_id = -1

    val_class_recall_named = {}
    for k, v in per_class_stats["val_class_recall_per_class"].items():
        if int(k) in inv_label_map:
            val_class_recall_named[inv_label_map[int(k)]] = v

    replay_class_ids_used = []
    for k in replay_class_counts_used.keys():
        replay_class_ids_used.append(int(k))

    metrics = MetricRecord({
        "client_id": actual_client_id,
        "server_round": server_round,
        "train_loss": train_loss,
        "val_loss": eval_metrics["loss"],
        "val_accuracy": eval_metrics["accuracy"],
        "val_macro_f1": eval_metrics["macro_f1"],
        "val_weighted_f1": eval_metrics["weighted_f1"],
        "num-examples": len(trainloader.dataset),
        "fit_duration_sec": fit_duration,
        "local_eval_duration_sec": local_eval_duration,
        "chunk_start_idx": chunk_start_idx,
        "chunk_end_idx": chunk_end_idx,
        "chunk_class_counts": chunk_class_counts_list,
        "chunk_present_class_ids": chunk_present_class_ids,
        "chunk_majority_class_id": chunk_majority_class_id,
        "global_model_total_l2": global_model_summary["global_model_total_l2"],
        "num_present_classes_in_chunk": len(chunk_present_class_ids),
        "num_absent_classes_restored": 0,
        "absent_class_ids_restored": [],
        "val_class_counts": val_class_counts_list,
        "val_present_class_ids": val_present_class_ids,
        "val_majority_class_id": val_majority_class_id,
        "replay_windows_used": replay_windows_used,
        "replay_buffer_size_after_update": replay_buffer_size_after_update,
        "num_train_examples_actual": len(current_y),
        "num_train_examples_with_replay": len(y_train_mix),
        "replay_missing_class_windows_used": replay_missing_class_windows_used,
        "replay_class_ids_used": replay_class_ids_used,
    })

    str_record = ConfigRecord({
        "replay_class_counts_used": json.dumps(replay_class_counts_used),
        "val_class_recall_per_class_named": json.dumps(val_class_recall_named),
    })

    state_to_send = model.state_dict()

    if personalization_mode == "fedper":
        state_to_send = split_shared_state(state_to_send)

    model_record = ArrayRecord.from_torch_state_dict(state_to_send)

    return Message(
        content=RecordDict({"arrays": model_record, "metrics": metrics, "str_metrics": str_record}),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg, context):
    partition_id = context.node_config["partition-id"]

    window_seconds = float(context.run_config["window-seconds"])
    overlap_ratio = float(context.run_config["overlap-ratio"])
    batch_size = int(context.run_config["batch-size"])
    precomputed_dir = str(context.run_config["precomputed-dir"])
    server_round = msg.content["config"]["server-round"]
    train_client_ids = parse_train_client_ids(msg.content["config"]["train-client-ids"])
    model_name = msg.content["config"]["model_name"]
    actual_client_id = int(train_client_ids[partition_id])
    channel_config = str(context.run_config["channel-config"])

    _, valloader, num_channels, num_classes, _, _, _ = load_client_data(
        partition_id=partition_id,
        window_seconds=window_seconds,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
        precomputed_dir=precomputed_dir,
        selected_client_ids=train_client_ids,
        channel_config=channel_config,
    )

    window_size = int(window_seconds * 20)
    model = get_model(model_name, num_channels, num_classes, window_size)
    state = msg.content["arrays"].to_torch_state_dict()
    personalization_mode = str(msg.content["config"]["personalization_mode"])

    if personalization_mode == "fedper":
        model.load_state_dict(state, strict=False)
        head_path = get_personal_head_path(precomputed_dir, actual_client_id)
        if os.path.exists(head_path):
            personal_state = torch.load(head_path, map_location=DEVICE)
            model_state = model.state_dict()
            model_state.update(personal_state)
            model.load_state_dict(model_state, strict=True)
    else:
        model.load_state_dict(state, strict=True)

    model.to(DEVICE)

    t0 = time.perf_counter()
    eval_metrics = evaluate_fn(model=model, dataloader=valloader, device=DEVICE)
    eval_duration = time.perf_counter() - t0

    metrics = MetricRecord({
        "client_id": actual_client_id,
        "server_round": server_round,
        "val_loss": eval_metrics["loss"],
        "val_accuracy": eval_metrics["accuracy"],
        "val_macro_f1": eval_metrics["macro_f1"],
        "val_weighted_f1": eval_metrics["weighted_f1"],
        "num-examples": eval_metrics["num_examples"],
        "eval_duration_sec": eval_duration,
    })

    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)