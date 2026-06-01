import json
from datetime import datetime
import os

def make_run_dir(prefix, output_root="outputs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, prefix + "_" + ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_json(data, path):
    f = open(path, "w")
    json.dump(data, f, indent=2)
    f.close()

def save_dataframe(df, path):
    df.to_csv(path, index=False)