import argparse
import json
import os

import pandas as pd
import torch

from harflwr.data_precomputed import invert_label_mapping, load_client_test_data
from harflwr.experiment_utils import save_dataframe
from harflwr.task import evaluate_fn, get_model


def main(run_dir, precomputed_dir_override=None):
    config_path = os.path.join(run_dir, "config.json")
    model_path = os.path.join(run_dir, "final_model.pt")

    f = open(config_path, "r")
    cfg = json.load(f)
    f.close()

    if precomputed_dir_override != None:
        precomputed_dir = precomputed_dir_override
    else:
        precomputed_dir = cfg["precomputed_dir"]

    batch_size = cfg["batch_size"]
    num_channels = cfg["num_channels"]
    num_classes = cfg["num_classes"]
    label_map = cfg["label_map"]
    inv_label_map = invert_label_mapping(label_map)
    window_size = cfg["window_size"]
    test_client_ids = cfg["test_client_ids"]
    model_name = cfg["model_name"]
    channel_config = cfg["channel_config"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = get_model(model_name, num_channels, num_classes, window_size)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    client_rows = []
    prediction_rows = []

    for client_id in test_client_ids:
        testloader, _, _, _, _, _ = load_client_test_data(
            partition_id=client_id,
            batch_size=batch_size,
            precomputed_dir=precomputed_dir,
            selected_client_ids=test_client_ids,
            channel_config=channel_config,
        )

        metrics = evaluate_fn(model, testloader, device)

        client_rows.append({
            "client_id": client_id,
            "num_test_examples": metrics["num_examples"],
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
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
        "channel_config": channel_config,
    }])

    save_dataframe(df_clients, os.path.join(run_dir, "per_client_metrics_test.csv"))
    save_dataframe(df_summary, os.path.join(run_dir, "final_eval_summary_metrics_test.csv"))
    save_dataframe(df_predictions, os.path.join(run_dir, "predictions_test.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--precomputed-dir", type=str, default=None)
    args = parser.parse_args()

    main(args.run_dir, args.precomputed_dir)