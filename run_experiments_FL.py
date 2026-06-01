import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

PYPROJECT = "pyproject.toml"
OUTPUT_DIR = "outputs"
REPS = 1
LEARNING_RATE = 0.0001

PRECOMPUTED_PARTA_BLOCKA = "precomputed/partA_blockA_magstd"
PRECOMPUTED_PARTA_BLOCKB = "precomputed/partA_blockB_magstd"
PRECOMPUTED_PARTB_BLOCKA = "precomputed/partB_blockA_magstd"
PRECOMPUTED_PARTB_BLOCKB = "precomputed/partB_blockB_magstd"

SUPERNODES = {
    PRECOMPUTED_PARTA_BLOCKA: 112,
    PRECOMPUTED_PARTA_BLOCKB: 115,
    PRECOMPUTED_PARTB_BLOCKA: 113,
    PRECOMPUTED_PARTB_BLOCKB: 113,
}

BLOCK_LABEL = {
    PRECOMPUTED_PARTA_BLOCKA: "partA_blockA",
    PRECOMPUTED_PARTA_BLOCKB: "partA_blockB",
    PRECOMPUTED_PARTB_BLOCKA: "partB_blockA",
    PRECOMPUTED_PARTB_BLOCKB: "partB_blockB",
}


def make_experiments():
    exps = []
    for precomputed_dir in [
        PRECOMPUTED_PARTA_BLOCKA,
        PRECOMPUTED_PARTA_BLOCKB,
        PRECOMPUTED_PARTB_BLOCKA,
        PRECOMPUTED_PARTB_BLOCKB,
    ]:
        block = BLOCK_LABEL[precomputed_dir]
        for channel_config in ["mag_std"]:
            cfg = channel_config.replace("_", "")
            for model in ["cnn", "mlp"]:
                # no replay
                exps.append({
                    "name": block + "_" + cfg + "_" + model,
                    "precomputed_dir": precomputed_dir,
                    "channel_config": channel_config,
                    "model-name": model,
                    "use_replay": False,
                    "personalization_mode": "none",
                })
                # fedper
                exps.append({
                    "name": block + "_" + cfg + "_fedper_" + model,
                    "precomputed_dir": precomputed_dir,
                    "channel_config": channel_config,
                    "model-name": model,
                    "use_replay": False,
                    "personalization_mode": "fedper",
                })
                # replay
                exps.append({
                    "name": block + "_" + cfg + "_replay_" + model,
                    "precomputed_dir": precomputed_dir,
                    "channel_config": channel_config,
                    "model-name": model,
                    "use_replay": True,
                    "personalization_mode": "none",
                })
    return exps


EXPERIMENTS = make_experiments()


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


def collect_reps(rep_dirs, experiment_name):
    dest = os.path.join(OUTPUT_DIR, experiment_name)
    os.makedirs(dest, exist_ok=True)
    for rep_dir in rep_dirs:
        run_name = os.path.basename(rep_dir)
        shutil.move(rep_dir, os.path.join(dest, run_name))
    print("\nCollected " + str(len(rep_dirs)) + " reps to " + dest)


def update_toml(exp):
    content = read_toml()
    num_supernodes = SUPERNODES[exp["precomputed_dir"]]
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


def setup():
    first_dir = EXPERIMENTS[0]["precomputed_dir"]
    num_supernodes = SUPERNODES[first_dir]
    print("  Enabling federation block (num_supernodes=" + str(num_supernodes) + ")...")
    content = read_toml()
    content = enable_federation_block(content, num_supernodes)
    write_toml(content)
    print("  Running pip install -e . ...")
    run_command(["pip", "install", "-e", ".", "-q"])
    print("  Setup complete")


def run_experiment(exp):
    name = exp["name"]
    print("\n" + "=" * 60)
    print("  EXPERIMENT: " + name)
    print("  precomputed_dir: " + exp["precomputed_dir"])
    print("  channel_config: " + exp["channel_config"])
    print("  model: " + exp["model-name"])
    print("  replay: " + str(exp["use_replay"]))
    print("  personalization: " + exp["personalization_mode"])
    print("  learning_rate: " + str(LEARNING_RATE))
    print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    rep_dirs = []
    for rep in range(1, REPS + 1):
        print("\n  --- Rep " + str(rep) + "/" + str(REPS) + " ---")
        update_toml(exp)
        run_command(["flwr", "run", ".", "--stream"])
        latest = get_latest_fl_run()
        run_command([
            "python", "-m", "harflwr.evaluate_fl",
            "--run-dir", latest,
            "--channel-config", exp["channel_config"],
        ])
        rep_dirs.append(latest)

    collect_reps(rep_dirs, name)
    print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def main(start_from=0, fedper_only=False):
    experiments = EXPERIMENTS
    if fedper_only:
        filtered = []
        for e in EXPERIMENTS:
            if e["personalization_mode"] == "fedper":
                filtered.append(e)
        experiments = filtered
        print("  FedPer-only mode: " + str(len(experiments)) + " experiments selected")

    total_runs = len(experiments) * REPS
    print("\n" + "=" * 60)
    print("  Full ablation — Part A + Part B, Block A + Block B")
    print("  Configs: std, mag_std")
    print("  Conditions: fedper only (CNN + MLP)")
    print("  Learning rate: " + str(LEARNING_RATE))
    print("  " + str(len(experiments)) + " experiments x " + str(REPS) + " reps = " + str(total_runs) + " FL runs")
    print("  Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")

    setup()

    for i in range(len(experiments)):
        exp = experiments[i]
        if i < start_from:
            print("[" + str(i + 1) + "/" + str(len(experiments)) + "] Skipping " + exp["name"])
            continue
        print("\n[" + str(i + 1) + "/" + str(len(experiments)) + "]", end=" ")
        run_experiment(exp)

    print("\n" + "=" * 60)
    print("  ALL DONE — " + str(len(experiments)) + " experiments complete")
    print("  Finished: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--fedper-only", action="store_true")
    args = parser.parse_args()
    main(start_from=args.start_from, fedper_only=args.fedper_only)