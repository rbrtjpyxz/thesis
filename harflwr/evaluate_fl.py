import argparse
import json
import os

import pandas as pd
import torch

from harflwr.data_precomputed import invert_label_mapping, load_client_test_data
from harflwr.experiment_utils import save_dataframe
from harflwr.task import evaluate_fn, get_model


def main(run_dir, data_dir_override=None, channel_config_override=None):
    config_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "final_model.pt")

    f = open(config_path, "r")
    cfg = json.load(f)
    f.close()

    personalization_mode = cfg["personalization_mode"]
    fedper_adapt_epochs = cfg["fedper_adapt_epochs"]

    if data_dir_override != None:
        precomputed_dir = data_dir_override
    else:
        precomputed_dir = cfg["precomputed_dir"]

    if channel_config_override != None:
        channel_config = channel_config_override
    else:
        channel_config = cfg["channel_config"]

    window_seconds = cfg["window_seconds"]
    overlap_ratio = cfg["overlap_ratio"]
    batch_size = cfg["batch_size"]
    label_map = cfg["label_map"]
    inv_label_map = invert_label_mapping(label_map)
    test_client_ids = cfg["test_client_ids"]
    model_name = cfg["model_name"]
    num_channels = cfg["num_channels"]
    num_classes = cfg["num_classes"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    state_dict = torch.load(model_path, map_location=device)

    client_rows = []
    prediction_rows = []

    for client_id in test_client_ids:
        testloader, _, _, _, _, _ = load_client_test_data(
            partition_id=client_id,
            window_seconds=window_seconds,
            overlap_ratio=overlap_ratio,
            batch_size=batch_size,
            precomputed_dir=precomputed_dir,
            selected_client_ids=test_client_ids,
            channel_config=channel_config,
        )

        window_size = int(window_seconds * 20)
        model = get_model(model_name, num_channels, num_classes, window_size)

        if personalization_mode == "fedper":
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            metrics = evaluate_fn(model, testloader, device)
            adapt_epochs = fedper_adapt_epochs
        else:
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            metrics = evaluate_fn(model, testloader, device)
            adapt_epochs = 0

        client_rows.append({
            "client_id": client_id,
            "num_test_examples": metrics["num_examples"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "personalization_mode": personalization_mode,
            "fedper_adapt_epochs": adapt_epochs,
        })

        for yt, yp in zip(metrics["y_true"], metrics["y_pred"]):
            prediction_rows.append({
                "client_id": client_id,
                "y_true": yt,
                "y_pred": yp,
                "y_true_label": inv_label_map[int(yt)],
                "y_pred_label": inv_label_map[int(yp)],
            })

    df_clients = pd.DataFrame(client_rows)
    df_predictions = pd.DataFrame(prediction_rows)

    df_summary = pd.DataFrame([{
        "mean_loss": df_clients["loss"].mean(),
        "std_loss": df_clients["loss"].std(),
        "mean_accuracy": df_clients["accuracy"].mean(),
        "std_accuracy": df_clients["accuracy"].std(),
        "mean_macro_f1": df_clients["macro_f1"].mean(),
        "std_macro_f1": df_clients["macro_f1"].std(),
        "mean_weighted_f1": df_clients["weighted_f1"].mean(),
        "std_weighted_f1": df_clients["weighted_f1"].std(),
        "num_clients": len(df_clients),
        "num_predictions": len(df_predictions),
        "evaluation_data_dir": precomputed_dir,
        "personalization_mode": personalization_mode,
    }])

    save_dataframe(df_clients, os.path.join(run_dir, "per_client_metrics_test.csv"))
    save_dataframe(df_summary, os.path.join(run_dir, "final_eval_summary_metrics_test.csv"))
    save_dataframe(df_predictions, os.path.join(run_dir, "predictions_test.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--channel-config", type=str, default=None)
    args = parser.parse_args()

    main(args.run_dir, args.data_dir, args.channel_config)