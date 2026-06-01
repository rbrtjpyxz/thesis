import json
import shutil
import os

import pandas as pd
import torch
from flwr.app import ArrayRecord, ConfigRecord, MetricRecord
from flwr.serverapp import ServerApp

from harflwr.customized_strategies import SequentialChunkFedAvg
from harflwr.data_precomputed import (
    compute_client_total_chunks,
    get_train_test_client_ids,
    load_client_data,
)
from harflwr.experiment_utils import make_run_dir, save_dataframe, save_json
from harflwr.task import get_model

app = ServerApp()


@app.main()
def main(grid, context):

    precomputed_dir = str(context.run_config["precomputed-dir"])
    window_seconds = float(context.run_config["window-seconds"])
    overlap_ratio = float(context.run_config["overlap-ratio"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    chunk_size = int(context.run_config["chunk-size"])
    model_name = str(context.run_config["model-name"])
    channel_config = str(context.run_config["channel-config"])

    personalization_mode = str(context.run_config["personalization-mode"])
    fedper_adapt_epochs = int(context.run_config["fedper-adapt-epochs"])

    use_replay_buffer = bool(context.run_config["use-replay-buffer"])
    replay_buffer_capacity = int(context.run_config["replay-buffer-capacity"])
    replay_sample_size = int(context.run_config["replay-sample-size"])

    replay_dir = os.path.join(precomputed_dir, "_replay_buffers")
    if os.path.exists(replay_dir):
        shutil.rmtree(replay_dir)

    personal_dir = os.path.join(precomputed_dir, "_personal_heads")
    if personalization_mode == "fedper":
        if os.path.exists(personal_dir):
            shutil.rmtree(personal_dir)

    train_client_ids, test_client_ids = get_train_test_client_ids(
        precomputed_dir=precomputed_dir,
        test_ratio=0.2,
        seed=42,
    )

    _, _, num_channels, num_classes, sensor_cols, label_map, _ = load_client_data(
        partition_id=0,
        window_seconds=window_seconds,
        overlap_ratio=overlap_ratio,
        batch_size=batch_size,
        precomputed_dir=precomputed_dir,
        selected_client_ids=train_client_ids,
        channel_config=channel_config,
    )
    num_classes = len(label_map)

    inv_label_map = {}
    for k, v in label_map.items():
        inv_label_map[int(v)] = k

    client_total_chunks = compute_client_total_chunks(
        train_client_ids=train_client_ids,
        precomputed_dir=precomputed_dir,
        chunk_size=chunk_size,
    )
    num_rounds = max(client_total_chunks.values())

    window_size = int(window_seconds * 20)
    model = get_model(model_name, num_channels, num_classes, window_size)

    full_state = model.state_dict()

    state_to_send = {}
    if personalization_mode == "fedper":
        for k, v in full_state.items():
            if not k.startswith("classifier."):
                state_to_send[k] = v.detach().cpu().clone()
    else:
        for k, v in full_state.items():
            state_to_send[k] = v.detach().cpu().clone()

    arrays = ArrayRecord.from_torch_state_dict(state_to_send)

    run_dir = make_run_dir(prefix="fl_run")

    if use_replay_buffer:
        replay_buffer_capacity_to_save = replay_buffer_capacity
        replay_sample_size_to_save = replay_sample_size
    else:
        replay_buffer_capacity_to_save = None
        replay_sample_size_to_save = None

    client_total_chunks_str = {}
    for k, v in client_total_chunks.items():
        client_total_chunks_str[str(k)] = v

    save_json(
        {
            "mode": "federated",
            "split_mode": "inter_subject",
            "precomputed_dir": precomputed_dir,
            "window_seconds": window_seconds,
            "overlap_ratio": overlap_ratio,
            "batch_size": batch_size,
            "num_server_rounds": num_rounds,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "fraction_train": fraction_train,
            "fraction_evaluate": fraction_evaluate,
            "channel_config": channel_config,
            "chunk_size": chunk_size,
            "model_name": model_name,
            "data_serving_mode": "sequential_chunks",
            "num_clients_total": len(train_client_ids) + len(test_client_ids),
            "num_train_clients": len(train_client_ids),
            "num_test_clients": len(test_client_ids),
            "train_client_ids": train_client_ids,
            "test_client_ids": test_client_ids,
            "num_channels": num_channels,
            "num_classes": num_classes,
            "sensor_cols": sensor_cols,
            "label_map": label_map,
            "client_total_chunks": client_total_chunks_str,
            "use_replay_buffer": use_replay_buffer,
            "replay_buffer_capacity": replay_buffer_capacity_to_save,
            "replay_sample_size": replay_sample_size_to_save,
            "aggregation_strategy": "FedAvg",
            "fedper_adapt_epochs": fedper_adapt_epochs,
            "personalization_mode": personalization_mode,
        },
        os.path.join(run_dir, "config.json"),
    )

    train_client_rows = []
    eval_client_rows = []

    client_selection_counts = {}
    for cid in train_client_ids:
        client_selection_counts[cid] = 0

    exhausted_client_ids = []

    def aggregate_train_metrics(records, weighting_key):
        total_weight = 0.0
        weighted_train_loss = 0.0
        weighted_fit_dur = 0.0
        weighted_eval_dur = 0.0

        for record in records:
            metrics = record["metrics"]
            weight = metrics["num-examples"]
            n_examples = metrics["num-examples"]
            total_weight += weight

            client_id = int(metrics["client_id"])
            server_round = metrics["server_round"]

            if n_examples > 0:
                if client_id in client_selection_counts:
                    client_selection_counts[client_id] = client_selection_counts[client_id] + 1
                else:
                    client_selection_counts[client_id] = 1

                if client_selection_counts[client_id] >= client_total_chunks[client_id]:
                    exhausted_client_ids.append(client_id)
                    print("Client " + str(client_id) + " exhausted after " +
                          str(client_selection_counts[client_id]) + " selections" +
                          " (" + str(client_total_chunks[client_id]) + " total chunks)")

            chunk_present_class_ids = metrics.get("chunk_present_class_ids", [])
            chunk_present_class_names = []
            for i in chunk_present_class_ids:
                if int(i) in inv_label_map:
                    chunk_present_class_names.append(inv_label_map[int(i)])

            chunk_majority_class_id = int(metrics.get("chunk_majority_class_id", -1))
            if chunk_majority_class_id in inv_label_map:
                chunk_majority_class_name = inv_label_map[chunk_majority_class_id]
            else:
                chunk_majority_class_name = "none"

            val_present_class_ids = metrics.get("val_present_class_ids", [])
            val_present_class_names = []
            for i in val_present_class_ids:
                if int(i) in inv_label_map:
                    val_present_class_names.append(inv_label_map[int(i)])

            val_majority_class_id = int(metrics.get("val_majority_class_id", -1))
            if val_majority_class_id in inv_label_map:
                val_majority_class_name = inv_label_map[val_majority_class_id]
            else:
                val_majority_class_name = "none"

            absent_class_ids_restored = metrics.get("absent_class_ids_restored", [])
            absent_class_names_restored = []
            for i in absent_class_ids_restored:
                if int(i) in inv_label_map:
                    absent_class_names_restored.append(inv_label_map[int(i)])

            replay_class_ids_used = metrics.get("replay_class_ids_used", [])
            replay_class_names_used = []
            for k in replay_class_ids_used:
                if int(k) in inv_label_map:
                    replay_class_names_used.append(inv_label_map[int(k)])

            if "str_metrics" in record:
                str_metrics = record["str_metrics"]
            else:
                str_metrics = ConfigRecord({})

            train_client_rows.append({
                "server_round": server_round,
                "client_id": client_id,
                "num_train_examples": n_examples,
                "train_loss": metrics["train_loss"],
                "fit_duration_sec": metrics["fit_duration_sec"],
                "local_eval_duration_sec": metrics["local_eval_duration_sec"],
                "train_val_loss": metrics["val_loss"],
                "train_val_accuracy": metrics["val_accuracy"],
                "train_val_macro_f1": metrics["val_macro_f1"],
                "train_val_weighted_f1": metrics["val_weighted_f1"],
                "chunk_start_idx": metrics.get("chunk_start_idx", -1),
                "chunk_end_idx": metrics.get("chunk_end_idx", -1),
                "chunk_class_counts": str(list(metrics.get("chunk_class_counts", []))),
                "chunk_present_class_ids": str(list(chunk_present_class_ids)),
                "chunk_present_class_names": str(chunk_present_class_names),
                "chunk_majority_class_id": chunk_majority_class_id,
                "chunk_majority_class_name": chunk_majority_class_name,
                "num_present_classes_in_chunk": metrics.get("num_present_classes_in_chunk", 0),
                "global_model_total_l2": metrics.get("global_model_total_l2", float("nan")),
                "val_class_counts": str(list(metrics.get("val_class_counts", []))),
                "val_present_class_ids": str(list(val_present_class_ids)),
                "val_present_class_names": str(val_present_class_names),
                "val_majority_class_id": val_majority_class_id,
                "val_majority_class_name": val_majority_class_name,
                "replay_windows_used": metrics.get("replay_windows_used", 0),
                "replay_buffer_size_after_update": metrics.get("replay_buffer_size_after_update", 0),
                "num_train_examples_actual": metrics.get("num_train_examples_actual", 0),
                "num_train_examples_with_replay": metrics.get("num_train_examples_with_replay", 0),
                "replay_missing_class_windows_used": metrics.get("replay_missing_class_windows_used", 0),
                "replay_class_counts_used": str(str_metrics.get("replay_class_counts_used", "{}")),
                "replay_class_ids_used": str(list(replay_class_ids_used)),
                "replay_class_names_used": str(replay_class_names_used),
                "val_class_recall_per_class_named": str(str_metrics.get("val_class_recall_per_class_named", "{}")),
                "num_absent_classes_restored": metrics.get("num_absent_classes_restored", 0),
                "absent_class_ids_restored": str(list(absent_class_ids_restored)),
                "absent_class_names_restored": str(absent_class_names_restored),
            })

            weighted_train_loss += metrics["train_loss"] * weight
            weighted_fit_dur += metrics["fit_duration_sec"] * weight
            weighted_eval_dur += metrics["local_eval_duration_sec"] * weight

        out = MetricRecord()
        if total_weight > 0:
            out["train_loss"] = weighted_train_loss / total_weight
            out["fit_duration_sec"] = weighted_fit_dur / total_weight
            out["local_eval_duration_sec"] = weighted_eval_dur / total_weight
        out["num-examples"] = total_weight
        return out

    def aggregate_eval_metrics(records, weighting_key):
        total_weight = 0.0
        weighted_val_loss = 0.0
        weighted_val_acc = 0.0
        weighted_val_macro = 0.0
        weighted_val_weighted = 0.0
        weighted_eval_dur = 0.0

        for record in records:
            metrics = record["metrics"]
            weight = 1.0
            total_weight += weight

            client_id = int(metrics["client_id"])
            server_round = metrics["server_round"]

            eval_client_rows.append({
                "server_round": server_round,
                "client_id": client_id,
                "num_val_examples": metrics["num-examples"],
                "val_loss": metrics["val_loss"],
                "val_accuracy": metrics["val_accuracy"],
                "val_macro_f1": metrics["val_macro_f1"],
                "val_weighted_f1": metrics["val_weighted_f1"],
                "eval_duration_sec": metrics["eval_duration_sec"],
            })

            weighted_val_loss += metrics["val_loss"] * weight
            weighted_val_acc += metrics["val_accuracy"] * weight
            weighted_val_macro += metrics["val_macro_f1"] * weight
            weighted_val_weighted += metrics["val_weighted_f1"] * weight
            weighted_eval_dur += metrics["eval_duration_sec"] * weight

        out = MetricRecord()
        if total_weight > 0:
            out["val_loss"] = weighted_val_loss / total_weight
            out["val_accuracy"] = weighted_val_acc / total_weight
            out["val_macro_f1"] = weighted_val_macro / total_weight
            out["val_weighted_f1"] = weighted_val_weighted / total_weight
            out["eval_duration_sec"] = weighted_eval_dur / total_weight
        out["num-examples"] = total_weight
        out["num_eval_clients"] = total_weight
        return out

    strategy = SequentialChunkFedAvg(
        selection_counts=client_selection_counts,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        weighted_by_key="num-examples",
        train_metrics_aggr_fn=aggregate_train_metrics,
        evaluate_metrics_aggr_fn=aggregate_eval_metrics,
    )

    client_total_chunks_str_config = {}
    for k, v in client_total_chunks.items():
        client_total_chunks_str_config[str(k)] = v

    train_config = ConfigRecord({
        "local-epochs": local_epochs,
        "learning-rate": learning_rate,
        "server-round": 0,
        "train-client-ids": ",".join(map(str, train_client_ids)),
        "num_classes": num_classes,
        "client-selection-counts": json.dumps(client_selection_counts),
        "client-total-chunks": json.dumps(client_total_chunks_str_config),
        "use_replay_buffer": int(use_replay_buffer),
        "replay_buffer_capacity": int(replay_buffer_capacity),
        "replay_sample_size": int(replay_sample_size),
        "model_name": model_name,
        "label_map": json.dumps(label_map),
        "personalization_mode": personalization_mode,
        "fedper_adapt_epochs": fedper_adapt_epochs,
    })

    evaluate_config = ConfigRecord({
        "server-round": 0,
        "train-client-ids": ",".join(map(str, train_client_ids)),
        "model_name": model_name,
        "client-total-chunks": json.dumps(client_total_chunks_str_config),
        "personalization_mode": personalization_mode,
    })

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        evaluate_config=evaluate_config,
        num_rounds=num_rounds,
    )

    train_metrics_by_round = getattr(result, "train_metrics_clientapp", {})
    eval_metrics_by_round = getattr(result, "evaluate_metrics_clientapp", {})
    n_clients = len(train_client_ids)

    history_rows = []
    for server_round in range(1, num_rounds + 1):
        rt = train_metrics_by_round.get(server_round, {})
        re = eval_metrics_by_round.get(server_round, {})
        history_rows.append({
            "round": server_round,
            "n_clients_total": n_clients,
            "n_clients_train_selected": max(1, int(round(n_clients * fraction_train))),
            "n_clients_eval_selected": max(1, int(round(n_clients * fraction_evaluate))),
            "train_loss": rt.get("train_loss", float("nan")),
            "num_train_examples": rt.get("num-examples", 0),
            "fit_duration_sec": rt.get("fit_duration_sec", 0.0),
            "local_eval_duration_sec": rt.get("local_eval_duration_sec", 0.0),
            "val_loss": re.get("val_loss", float("nan")),
            "val_accuracy": re.get("val_accuracy", float("nan")),
            "val_macro_f1": re.get("val_macro_f1", float("nan")),
            "val_weighted_f1": re.get("val_weighted_f1", float("nan")),
            "num_val_examples": re.get("num-examples", 0),
            "num_eval_clients": re.get("num_eval_clients", 0),
            "eval_duration_sec": re.get("eval_duration_sec", 0.0),
        })

    history_df = pd.DataFrame(history_rows)
    save_dataframe(history_df, os.path.join(run_dir, "history.csv"))

    summary_df = history_df[[
        "round", "train_loss", "val_loss", "val_accuracy",
        "val_macro_f1", "val_weighted_f1",
        "num_train_examples", "num_val_examples",
        "fit_duration_sec", "local_eval_duration_sec", "eval_duration_sec",
    ]].tail(1).copy()
    save_dataframe(summary_df, os.path.join(run_dir, "summary_metrics.csv"))

    train_client_df = pd.DataFrame(train_client_rows)
    eval_client_df = pd.DataFrame(eval_client_rows)

    if len(train_client_rows) > 0 or len(eval_client_rows) > 0:
        if len(train_client_rows) == 0:
            per_client_round_df = eval_client_df.copy()
        elif len(eval_client_rows) == 0:
            per_client_round_df = train_client_df.copy()
        else:
            per_client_round_df = pd.merge(
                train_client_df, eval_client_df,
                on=["server_round", "client_id"],
                how="outer",
            )
        per_client_round_df = per_client_round_df.sort_values(["server_round", "client_id"]).reset_index(drop=True)
        save_dataframe(per_client_round_df, os.path.join(run_dir, "per_client_round_metrics.csv"))

    final_state_dict = {}
    for k, v in result.arrays.to_torch_state_dict().items():
        final_state_dict[k] = v.detach().cpu().clone()

    torch.save(final_state_dict, os.path.join(run_dir, "final_model.pt"))

    print("Run saved to: " + str(run_dir))
    print("Final model: " + os.path.join(run_dir, "final_model.pt"))
    print("History: " + os.path.join(run_dir, "history.csv"))
    print("Summary: " + os.path.join(run_dir, "summary_metrics.csv"))
    print("Per-client metrics: " + os.path.join(run_dir, "per_client_round_metrics.csv"))