import sys
import jsonpickle
import matplotlib.pyplot as plt
import os

def plot_metric_vs_model(model_names, metric_values, metric_name, output_path):
    plt.figure(figsize=(8, max(4, len(model_names) * 0.5)))
    y_pos = range(len(model_names))
    plt.scatter(metric_values, y_pos)
    plt.yticks(y_pos, model_names)
    plt.xlabel(metric_name)
    plt.ylabel("Model Name")
    plt.title(f"{metric_name} per Model")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot a metric vs model name from metrics files.")
    parser.add_argument("metrics_files", nargs="+", help="List of metrics.json files")
    parser.add_argument("--output", type=str, required=True, help="Output path for the plot (PNG)")
    parser.add_argument("--metric", type=str, default="min_val_loss", help="Metric to plot (default: min_val_loss)")
    args = parser.parse_args()

    model_names = []
    metric_values = []

    for metrics_file in args.metrics_files:
        with open(metrics_file, "r") as f:
            data = jsonpickle.decode(f.read())
        # Get model name from metadata if available
        model_name = data.get("metadata", {}).get("model_name")
        if not model_name:
            model_name = os.path.basename(os.path.dirname(metrics_file))
        model_names.append(model_name)
        metric_val = data["metrics"].get(args.metric)
        if metric_val is None:
            raise ValueError(f"Metric '{args.metric}' not found in {metrics_file}")
        metric_values.append(metric_val)

    plot_metric_vs_model(model_names, metric_values, args.metric, args.output)

if __name__ == "__main__":
    main()
