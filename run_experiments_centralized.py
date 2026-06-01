import os
import shutil
import subprocess
import sys
from datetime import datetime
import time

OUTPUT_DIR = "outputs"
PRECOMPUTED_DIR = "precomputed/partB_blockB_magstd"

LEARNING_RATES = [0.1, 0.001, 0.0001, 0.00001]
CHANNEL_CONFIGS = ["mag_std"]
MODEL_NAMES = ["cnn", "mlp"]
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
SEED = 42


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
    dest = os.path.join(OUTPUT_DIR, "c_results", experiment_name)
    os.makedirs(dest, exist_ok=True)
    run_name = os.path.basename(run_dir)
    shutil.move(run_dir, os.path.join(dest, run_name))
    print("\nSaved to " + os.path.join(dest, run_name))


def main():
    experiments = []
    for channel_config in CHANNEL_CONFIGS:
        for model_name in MODEL_NAMES:
            for lr in LEARNING_RATES:
                experiments.append({
                    "name": channel_config + "_" + model_name + "_lr" + str(lr),
                    "channel_config": channel_config,
                    "model_name": model_name,
                    "lr": lr,
                })

    total = len(experiments)
    print("\n" + "=" * 60)
    print("  Centralized LR ablation")
    print("  " + str(total) + " experiments")
    print("  LRs: " + str(LEARNING_RATES))
    print("  Configs: " + str(CHANNEL_CONFIGS))
    print("  Models: " + str(MODEL_NAMES))
    print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")

    for i in range(len(experiments)):
        exp = experiments[i]
        name = exp["name"]
        print("\n[" + str(i + 1) + "/" + str(total) + "] EXPERIMENT: " + name)
        print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        run_command([
            "python", "-m", "harflwr.centralized",
            "--precomputed-dir", PRECOMPUTED_DIR,
            "--channel-config", exp["channel_config"],
            "--model-name", exp["model_name"],
            "--learning-rate", str(exp["lr"]),
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
    print("  Results in: " + os.path.join(OUTPUT_DIR, "centralized_results"))
    print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()