import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

OUTPUT_DIR = "outputs"
CHANNEL_CONFIG = "mag_std"
LEARNING_RATE = 0.0001
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
SEED = 42

WINDOW_SIZES = [2, 4, 5, 8]
MODELS = ["cnn", "mlp"]

PRECOMPUTED = {
    ("A", 5): "precomputed/partA_blockB_magstd",
    ("A", 2): "precomputed/partA_blockB_magstd_2s",
    ("A", 4): "precomputed/partA_blockB_magstd_4s",
    ("A", 8): "precomputed/partA_blockB_magstd_8s",
    ("B", 5): "precomputed/partB_blockB_magstd",
    ("B", 2): "precomputed/partB_blockB_magstd_2s",
    ("B", 4): "precomputed/partB_blockB_magstd_4s",
    ("B", 8): "precomputed/partB_blockB_magstd_8s",
}


def run_command(cmd):
    print("\n>>> " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nCommand failed: " + " ".join(cmd))
        sys.exit(result.returncode)


def get_latest_centralized_run():
    all_items = os.listdir(OUTPUT_DIR)
    runs = []
    for item in all_items:
        if item.startswith("centralized_"):
            full_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(full_path):
                runs.append(full_path)
    runs = sorted(runs)
    return runs[-1]


def collect_run(run_dir, experiment_name):
    dest = os.path.join(OUTPUT_DIR, "window_results", experiment_name)
    os.makedirs(dest, exist_ok=True)
    run_name = os.path.basename(run_dir)
    shutil.move(run_dir, os.path.join(dest, run_name))
    print("\nSaved to " + os.path.join(dest, run_name))


def main(parts, window_sizes, models):
    experiments = []
    for part in parts:
        for ws in window_sizes:
            for model in models:
                experiments.append({
                    "name": "part" + part + "_blockB_" + CHANNEL_CONFIG + "_" + str(ws) + "s_" + model,
                    "part": part,
                    "window_seconds": ws,
                    "model": model,
                    "precomputed_dir": PRECOMPUTED[(part, ws)],
                })

    missing = []
    for e in experiments:
        if not os.path.exists(e["precomputed_dir"]):
            missing.append(e)

    if len(missing) > 0:
        print("\nMissing precomputed dirs — will be skipped:")
        for e in missing:
            print("     " + e["precomputed_dir"] + "  (" + e["name"] + ")")

    valid_experiments = []
    for e in experiments:
        if os.path.exists(e["precomputed_dir"]):
            valid_experiments.append(e)
    experiments = valid_experiments

    total = len(experiments)
    print("\n" + "=" * 60)
    print("  Window size ablation — centralized baseline")
    print("  Channel config: " + CHANNEL_CONFIG)
    print("  Learning rate: " + str(LEARNING_RATE))
    print("  Parts: " + str(parts))
    print("  Window sizes: " + str(window_sizes) + "s")
    print("  Models: " + str(models))
    print("  Total runs: " + str(total))
    print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")

    for i in range(len(experiments)):
        exp = experiments[i]
        name = exp["name"]
        print("\n[" + str(i + 1) + "/" + str(total) + "] EXPERIMENT: " + name)
        print("  precomputed_dir: " + exp["precomputed_dir"])
        print("  window_seconds: " + str(exp["window_seconds"]) + "s")
        print("  model: " + exp["model"])
        print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        run_command([
            "python", "-m", "harflwr.centralized",
            "--precomputed-dir", exp["precomputed_dir"],
            "--channel-config", CHANNEL_CONFIG,
            "--model-name", exp["model"],
            "--learning-rate", str(LEARNING_RATE),
            "--seed", str(SEED),
        ])

        time.sleep(2)
        latest = get_latest_centralized_run()

        run_command([
            "python", "-m", "harflwr.evaluate_centralized",
            "--run-dir", latest,
        ])

        collect_run(latest, name)
        print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("\n" + "=" * 60)
    print("  ALL DONE — " + str(total) + " experiments complete")
    print("  Results in: " + os.path.join(OUTPUT_DIR, "window_results"))
    print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="both")
    parser.add_argument("--model", default="both")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=WINDOW_SIZES)
    args = parser.parse_args()

    if args.part == "both":
        parts = ["A", "B"]
    else:
        parts = [args.part]

    if args.model == "both":
        models = MODELS
    else:
        models = [args.model]

    main(parts=parts, window_sizes=args.window_sizes, models=models)