import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

PYPROJECT = "pyproject.toml"
OUTPUT_DIR = "outputs"
REPS = 5
LEARNING_RATE = 0.0001
CHANNEL_CONFIG = "mag_std"
MODELS = ["cnn","mlp"]
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 32
SEED = 42

DATASETS = {
    "FLAAP": ("precomputed/FLAAP_magstd", 6),
    "HAPT": ("precomputed/HAPT_magstd", 24),
    "HHAR": ("precomputed/HHAR_magstd", 7),
    "KU_HAR": ("precomputed/KU_HAR_magstd", 65),
    "RealWorld": ("precomputed/RealWorld_magstd", 12),
    "HARTH": ("precomputed/HARTH_magstd", 16),
    "WISDM_AP": ("precomputed/WISDM_AP_magstd", 28),
    "WISDM_Actitracker": ("precomputed/WISDM_Actitracker_magstd", 60),
    "UT_SAD": ("precomputed/UT_SAD_magstd", 8),
}

PART_A = ["FLAAP", "HAPT", "HHAR", "KU_HAR", "RealWorld"]
PART_B = ["HARTH", "WISDM_AP", "WISDM_Actitracker", "UT_SAD"]


def make_fl_experiments(precomputed_dir, dataset_name):
    exps = []
    cfg = CHANNEL_CONFIG.replace("_", "")
    for model in MODELS:
        for condition, use_replay, personalization in [
            ("baseline", False, "none"),
            ("fedper", False, "fedper"),
            ("replay", True, "none"),
        ]:
            exps.append({
                "name": dataset_name + "_" + cfg + "_" + condition + "_" + model,
                "precomputed_dir": precomputed_dir,
                "channel_config": CHANNEL_CONFIG,
                "model-name": model,
                "use_replay": use_replay,
                "personalization_mode": personalization,
            })
    return exps


def read_toml():
    f = open(PYPROJECT, "r")
    content = f.read()
    f.close()
    return content


def write_toml(content):
    f = open(PYPROJECT, "w")
    f.write(content)
    f.close()


def set_toml_value(content, key, value):
    pattern = r'^(' + re.escape(key) + r'\s*=\s*).*$'
    replacement = r'\g<1>' + value
    new_content, n = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    return new_content


def enable_federation_block(content, num_supernodes):
    content = content.replace(
        "# [tool.flwr.federations]\n"
        "# default = \"local-simulation\"\n"
        "\n"
        "# [tool.flwr.federations.local-simulation]\n"
        "# options.num-supernodes = ",
        "[tool.flwr.federations]\n"
        "default = \"local-simulation\"\n"
        "\n"
        "[tool.flwr.federations.local-simulation]\n"
        "options.num-supernodes = "
    )
    content = content.replace(
        "# address = \":local:\"\n"
        "# options.backend.client-resources.num-cpus = 1\n"
        "# options.backend.client-resources.num-gpus = 0.25",
        "address = \":local:\"\n"
        "options.backend.client-resources.num-cpus = 1\n"
        "options.backend.client-resources.num-gpus = 0.25"
    )
    content = re.sub(
        r'(options\.num-supernodes\s*=\s*)\d+',
        r'\g<1>' + str(num_supernodes),
        content
    )
    return content


def update_toml_fl(exp, num_supernodes):
    content = read_toml()
    content = enable_federation_block(content, num_supernodes)
    content = set_toml_value(content, "precomputed-dir", '"' + exp["precomputed_dir"] + '"')
    content = set_toml_value(content, "channel-config", '"' + exp["channel_config"] + '"')
    if exp["use_replay"]:
        content = set_toml_value(content, "use-replay-buffer", "true")
    else:
        content = set_toml_value(content, "use-replay-buffer", "false")
    content = set_toml_value(content, "personalization-mode", '"' + exp["personalization_mode"] + '"')
    content = set_toml_value(content, "model-name", '"' + exp["model-name"] + '"')
    content = set_toml_value(content, "learning-rate", str(LEARNING_RATE))
    write_toml(content)


def run_command(cmd):
    print("\n>>> " + " ".join(cmd) + "\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nCommand failed: " + " ".join(cmd))
        sys.exit(result.returncode)


def get_latest_fl_run():
    all_items = os.listdir(OUTPUT_DIR)
    runs = []
    for item in all_items:
        if item.startswith("fl_run_"):
            full_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(full_path):
                runs.append(full_path)
    runs = sorted(runs)
    return runs[-1]


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


def collect_to_dataset_folder(run_dir, dataset_name, subfolder):
    dest = os.path.join(OUTPUT_DIR, "per_dataset", dataset_name, subfolder)
    os.makedirs(dest, exist_ok=True)
    run_name = os.path.basename(run_dir)
    shutil.move(run_dir, os.path.join(dest, run_name))
    print("\nSaved to " + os.path.join(dest, run_name))


def collect_fl_reps(rep_dirs, dataset_name, exp_name):
    dest = os.path.join(OUTPUT_DIR, "per_dataset", dataset_name, exp_name)
    os.makedirs(dest, exist_ok=True)
    for rep_dir in rep_dirs:
        run_name = os.path.basename(rep_dir)
        shutil.move(rep_dir, os.path.join(dest, run_name))
    print("\nCollected " + str(len(rep_dirs)) + " reps to " + dest)


def run_dataset(dataset_name, precomputed_dir, num_supernodes):
    print("\n" + "#" * 60)
    print("  DATASET: " + dataset_name)
    print("  precomputed_dir: " + precomputed_dir)
    print("  num_supernodes: " + str(num_supernodes))
    print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 60 + "\n")

    run_command(["pip", "install", "-e", ".", "-q"])

    for model in MODELS:
        print("\n" + "=" * 60)
        print("  CENTRALIZED: " + dataset_name + " — " + CHANNEL_CONFIG + " — " + model)
        print("=" * 60)

        run_command([
            "python", "-m", "harflwr.centralized",
            "--precomputed-dir", precomputed_dir,
            "--channel-config", CHANNEL_CONFIG,
            "--model-name", model,
            "--learning-rate", str(LEARNING_RATE),
            "--seed", str(SEED),
        ])

        time.sleep(2)
        latest = get_latest_centralized_run()

        run_command([
            "python", "-m", "harflwr.evaluate_centralized",
            "--run-dir", latest,
        ])

        collect_to_dataset_folder(
            latest, dataset_name,
            "centralized_" + CHANNEL_CONFIG.replace("_", "") + "_" + model
        )

    fl_experiments = make_fl_experiments(precomputed_dir, dataset_name)

    for exp in fl_experiments:
        print("\n" + "=" * 60)
        print("  FL EXPERIMENT: " + exp["name"])
        print("  model: " + exp["model-name"])
        print("  replay: " + str(exp["use_replay"]))
        print("  personalization: " + exp["personalization_mode"])
        print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 60)

        rep_dirs = []
        for rep in range(1, REPS + 1):
            print("\n  --- Rep " + str(rep) + "/" + str(REPS) + " ---")
            update_toml_fl(exp, num_supernodes)
            run_command(["flwr", "run", ".", "--stream"])
            latest = get_latest_fl_run()
            run_command([
                "python", "-m", "harflwr.evaluate_fl",
                "--run-dir", latest,
                "--channel-config", exp["channel_config"],
            ])
            rep_dirs.append(latest)

        collect_fl_reps(rep_dirs, dataset_name, exp["name"])
        print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("\n" + "#" * 60)
    print("  DATASET DONE: " + dataset_name)
    print("  Results in: " + os.path.join(OUTPUT_DIR, "per_dataset", dataset_name))
    print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 60 + "\n")


def main(part="both", dataset_filter=None, start_from=0):
    if dataset_filter != None:
        selected = [dataset_filter]
    elif part == "A":
        selected = PART_A
    elif part == "B":
        selected = PART_B
    else:
        selected = PART_A + PART_B

    missing = []
    for d in selected:
        if not os.path.exists(DATASETS[d][0]):
            missing.append(d)

    if len(missing) > 0:
        print("\nMissing precomputed dirs — will be skipped:")
        for d in missing:
            print("     " + DATASETS[d][0] + "  (" + d + ")")

    valid_selected = []
    for d in selected:
        if d not in missing:
            valid_selected.append(d)
    selected = valid_selected

    total_datasets = len(selected)
    fl_per_dataset = len(MODELS) * 3
    total_fl_runs = total_datasets * fl_per_dataset * REPS
    total_c_runs = total_datasets * len(MODELS)

    print("\n" + "=" * 60)
    print("  Per-dataset experiment runner")
    print("  Channel config: " + CHANNEL_CONFIG)
    print("  Learning rate: " + str(LEARNING_RATE))
    print("  Datasets: " + str(selected))
    print("  Centralized: " + str(total_c_runs) + " runs (1 per model per dataset)")
    print("  FL: " + str(total_fl_runs) + " runs (" + str(fl_per_dataset) + " exps x " + str(REPS) + " reps)")
    print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")

    for i in range(len(selected)):
        dataset_name = selected[i]
        if i < start_from:
            print("[" + str(i + 1) + "/" + str(total_datasets) + "] Skipping " + dataset_name)
            continue
        print("\n[" + str(i + 1) + "/" + str(total_datasets) + "]", end=" ")
        precomputed_dir, num_supernodes = DATASETS[dataset_name]
        run_dataset(dataset_name, precomputed_dir, num_supernodes)

    print("\n" + "=" * 60)
    print("  ALL DONE — " + str(total_datasets) + " datasets complete")
    print("  Results in: " + os.path.join(OUTPUT_DIR, "per_dataset"))
    print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", default="both")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    main(
        part=args.part,
        dataset_filter=args.dataset,
        start_from=args.start_from,
    )