import argparse
import random
import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score

from harflwr.data_precomputed import (
    get_train_test_client_ids,
    get_client_filepaths_from_subset,
    invert_label_mapping,
    load_all_data_centralized_arrays,
    make_loader,
    read_metadata,
    select_channels,
)
from harflwr.experiment_utils import make_run_dir, save_dataframe, save_json
from harflwr.task import get_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(precomputed_dir="precomputed", channel_config="mag_std", model_name="cnn",
         learning_rate=0.001, seed=42):

    batch_size = 32
    max_epochs = 200
    patience = 20

    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: " + str(device))

    train_client_ids, test_client_ids = get_train_test_client_ids(
        precomputed_dir=precomputed_dir,
    )

    metadata = read_metadata(precomputed_dir)
    label_map = metadata["label_map"]
    sensor_cols = metadata["sensor_cols"]
    window_size = int(round(metadata["window_seconds"] * 20))
    num_classes = len(label_map)
    inv_label_map = invert_label_mapping(label_map)

    X_train_all, y_train_all, _, _ = load_all_data_centralized_arrays(
        precomputed_dir=precomputed_dir,
        selected_client_ids=train_client_ids,
    )

    X_val_parts = []
    y_val_parts = []
    for i in range(len(train_client_ids)):
        _, val_path = get_client_filepaths_from_subset(
            subset_client_ids=train_client_ids,
            subset_partition_id=i,
            precomputed_dir=precomputed_dir,
        )
        val_arr = np.load(val_path, allow_pickle=True)
        X_val = val_arr["X"]
        y_val = val_arr["y"]
        if len(X_val) > 0:
            X_val_parts.append(X_val)
            y_val_parts.append(y_val)

    X_val_all = np.concatenate(X_val_parts, axis=0)
    y_val_all = np.concatenate(y_val_parts, axis=0)

    if channel_config != "raw_mag_deriv":
        X_train_all, selected_cols = select_channels(X_train_all, sensor_cols, channel_config)
        X_val_all, _ = select_channels(X_val_all, sensor_cols, channel_config)
    else:
        selected_cols = sensor_cols

    num_channels = X_train_all.shape[1]

    trainloader = make_loader(X_train_all, y_train_all, batch_size=batch_size, shuffle=True, drop_last=True)
    valloader = make_loader(X_val_all, y_val_all, batch_size=batch_size, shuffle=False)

    print("Train windows: " + str(len(X_train_all)) + " | Val windows: " + str(len(X_val_all)))
    print("Channels: " + str(selected_cols) + " (" + str(num_channels) + ")")
    print("Classes: " + str(num_classes))

    model = get_model(model_name, num_channels, num_classes, window_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    run_dir = make_run_dir(prefix="centralized")
    cfg = {
        "mode": "centralized",
        "precomputed_dir": precomputed_dir,
        "channel_config": channel_config,
        "model_name": model_name,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "seed": seed,
        "num_channels": num_channels,
        "num_classes": num_classes,
        "window_size": window_size,
        "sensor_cols": selected_cols,
        "label_map": label_map,
        "num_train_clients": len(train_client_ids),
        "num_test_clients": len(test_client_ids),
        "train_client_ids": train_client_ids,
        "test_client_ids": test_client_ids,
    }

    best_val_macro_f1 = 0.0
    epochs_without_improve = 0
    best_model_state = None
    history_rows = []

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, max_epochs + 1):

        model.train()
        running_loss = 0.0
        total = 0
        for xb, yb in trainloader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            total += yb.size(0)
        train_loss = running_loss / total

        model.eval()
        val_loss = 0.0
        correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in valloader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                preds = torch.argmax(logits, dim=1)
                val_loss += loss.item() * yb.size(0)
                correct += (preds == yb).sum().item()
                val_total += yb.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())

        avg_val_loss = val_loss / val_total
        val_accuracy = correct / val_total
        current_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        if current_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_f1
            epochs_without_improve = 0
            best_model_state = {}
            for k, v in model.state_dict().items():
                best_model_state[k] = v.detach().cpu().clone()
        else:
            epochs_without_improve += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": current_f1,
            "val_weighted_f1": val_weighted_f1,
            "num_val_examples": val_total,
        })

        print("Epoch " + str(epoch) + "/" + str(max_epochs) +
              " | train_loss=" + str(round(train_loss, 4)) +
              " | val_loss=" + str(round(avg_val_loss, 4)) +
              " | val_macro_f1=" + str(round(current_f1, 4)) +
              " | best=" + str(round(best_val_macro_f1, 4)) +
              " | patience=" + str(epochs_without_improve) + "/" + str(patience))

        if epochs_without_improve >= patience:
            print("Early stopping at epoch " + str(epoch))
            break

    if best_model_state != None:
        model.load_state_dict(best_model_state)

    history_df = pd.DataFrame(history_rows)
    best_idx = history_df["val_macro_f1"].idxmax()

    cfg["best_epoch"] = int(best_idx) + 1
    cfg["best_val_macro_f1"] = best_val_macro_f1
    save_json(cfg, os.path.join(run_dir, "config.json"))

    save_dataframe(history_df, os.path.join(run_dir, "history.csv"))
    save_dataframe(history_df.loc[[best_idx]].copy(), os.path.join(run_dir, "summary_metrics.csv"))

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in valloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    preds_df = pd.DataFrame({
        "y_true": all_labels,
        "y_pred": all_preds,
        "y_true_label": [inv_label_map[int(y)] for y in all_labels],
        "y_pred_label": [inv_label_map[int(y)] for y in all_preds],
    })
    save_dataframe(preds_df, os.path.join(run_dir, "predictions.csv"))

    if best_model_state != None:
        final_state_dict = best_model_state
    else:
        final_state_dict = {}
        for k, v in model.state_dict().items():
            final_state_dict[k] = v.detach().cpu().clone()

    torch.save(final_state_dict, os.path.join(run_dir, "final_model.pt"))

    print("Run saved to: " + str(run_dir))
    print("Best epoch: " + str(cfg["best_epoch"]))
    print("Best val F1: " + str(round(best_val_macro_f1, 4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed-dir", type=str, default="precomputed")
    parser.add_argument("--channel-config", type=str, default="mag_std")
    parser.add_argument("--model-name", type=str, default="cnn")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        precomputed_dir=args.precomputed_dir,
        channel_config=args.channel_config,
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )