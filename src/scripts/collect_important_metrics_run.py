import wandb
import argparse
import os
import re
import jsonpickle
import pandas as pd

def _wandb_id_path(model_dir):
    return os.path.join(model_dir, "wandb_id.txt")

def _read_wandb_id(model_dir):
    wandb_id_path = _wandb_id_path(model_dir)
    if not os.path.isfile(wandb_id_path):
        return None
    with open(wandb_id_path, "r") as f:
        return f.read().strip()

def _get_local_history(model_dir):
    metrics_csv_path = os.path.join(model_dir, "metrics.csv")
    if not os.path.isfile(metrics_csv_path):
        raise FileNotFoundError(
            f"Neither wandb_id.txt nor metrics.csv found in {model_dir}. "
            "Training must be run with W&B enabled or with the local CSVLogger."
        )
    return pd.read_csv(metrics_csv_path)

def get_run_history(experiment_setting):
    model_dir = experiment_setting["model_dir"]
    entity = experiment_setting.get("entity")
    project = experiment_setting.get("project", "transit_project")
    run_id = _read_wandb_id(model_dir)
    if run_id is None:
        return _get_local_history(model_dir)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()
    return history

def get_min_val_loss(experiment_setting):
    history = get_run_history(experiment_setting)
    if 'val_loss' not in history.columns:
        raise ValueError("'val_loss' not found in the run history.")
    return float(history['val_loss'].min())

def get_min_val_loss_epoch(experiment_setting):
    history = get_run_history(experiment_setting)
    if 'val_loss' not in history.columns or 'epoch' not in history.columns:
        raise ValueError("'val_loss' or 'epoch' not found in the run history.")
    valid = history.dropna(subset=['val_loss'])
    min_idx = valid['val_loss'].idxmin()
    return int(valid.loc[min_idx, 'epoch'])

def get_metadata(experiment_setting):
    model_dir = experiment_setting["model_dir"]
    run_id = _read_wandb_id(model_dir)
    model_name = os.path.basename(os.path.normpath(model_dir))
    return {"wandb_id": run_id, "model_name": model_name}

def get_training_runtime(experiment_setting):
    model_dir = experiment_setting["model_dir"]
    entity = experiment_setting.get("entity")
    project = experiment_setting.get("project", "transit_project")
    run_id = _read_wandb_id(model_dir)
    if run_id is None:
        # No W&B run: fall back to the wall-clock time train_ptl.py writes locally.
        execution_time_path = os.path.join(model_dir, "execution_time.txt")
        if not os.path.isfile(execution_time_path):
            raise FileNotFoundError(f"execution_time.txt not found in {model_dir}")
        with open(execution_time_path, "r") as f:
            match = re.search(r"([\d.]+)\s*seconds", f.read())
        if not match:
            raise ValueError(f"Could not parse runtime from {execution_time_path}")
        return float(match.group(1))
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    # Try to get runtime from summary, fallback to duration if available
    runtime = run.summary.get("train_runtime")
    if runtime is None:
        runtime = run.summary.get("runtime")
    if runtime is None and hasattr(run, "duration"):
        runtime = run.duration
    if runtime is None:
        # Try to deduce from history
        history = run.history()
        # Try to use '_runtime' or '_timestamp' columns if present
        if '_runtime' in history.columns:
            runtime = history['_runtime'].max() - history['_runtime'].min()
            print(f"Using '_runtime' to calculate runtime: {runtime}")
        elif '_timestamp' in history.columns:
            runtime = history['_timestamp'].max() - history['_timestamp'].min()
            print(f"Using '_timestamp' to calculate runtime: {runtime}")
        else:
            raise ValueError("Training runtime not found in W&B run summary or history.")
    return float(runtime)

def get_gpu_utilization_percent(experiment_setting):
    model_dir = experiment_setting["model_dir"]
    entity = experiment_setting.get("entity")
    project = experiment_setting.get("project", "transit_project")
    run_id = _read_wandb_id(model_dir)
    if run_id is None:
        # No local equivalent to W&B's system metrics.
        print("No W&B run available. GPU utilization is not tracked locally. Returning None.")
        return None
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()
    # Try common GPU utilization keys
    gpu_keys = [
        "gpu/utilization", "gpu_utilization", "GPU Utilization", "system.gpu", "system/gpu/utilization"
    ]
    for key in gpu_keys:
        if key in history.columns:
            util = history[key].mean()
            print(f"Using '{key}' to calculate GPU utilization: {util}")
            return float(util)
    # Try to find any column containing 'gpu' and 'util'
    for col in history.columns:
        if 'gpu' in col.lower() and 'util' in col.lower():
            util = history[col].mean()
            print(f"Using '{col}' to calculate GPU utilization: {util}")
            return float(util)
    # If not found, return None
    print("GPU utilization not found in W&B run history. Returning None.")
    return None

def get_final_val_loss(experiment_setting):
    history = get_run_history(experiment_setting)
    if 'val_loss' not in history.columns:
        raise ValueError("'val_loss' not found in the run history.")
    return float(history['val_loss'].dropna().iloc[-1])

def collect_metrics(model_dir, entity=None, project="transit_project", output_path=None):
    experiment_setting = {
        "model_dir": model_dir,
        "entity": entity,
        "project": project
    }
    metrics = {
        "min_val_loss": get_min_val_loss(experiment_setting),
        "min_val_loss_epoch": get_min_val_loss_epoch(experiment_setting),
        "final_val_loss": get_final_val_loss(experiment_setting),
        "training_runtime": get_training_runtime(experiment_setting),
        "gpu_utilization_percent": get_gpu_utilization_percent(experiment_setting),
        # Add more metrics here as needed
    }
    metadata = get_metadata(experiment_setting)
    result = {
        "metrics": metrics,
        "metadata": metadata
    }
    if output_path is None:
        output_path = os.path.join(model_dir, "metrics.json")
    with open(output_path, "w") as f:
        f.write(jsonpickle.dumps(result))
    print(f"Metrics and metadata saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect W&B metrics and metadata from a model directory.")
    parser.add_argument("model_dir", type=str, help="Path to the folder with the trained model (must contain wandb_id.txt, or metrics.csv/execution_time.txt when W&B is skipped)")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (required only if training was run with a real W&B key, not with wandb_key=skip)")
    parser.add_argument("--project", type=str, default="transit_project", help="W&B project (default: transit_project)")
    parser.add_argument("--output", type=str, default=None, help="Output path for the metrics JSON file")
    args = parser.parse_args()

    collect_metrics(args.model_dir, args.entity, args.project, args.output)
