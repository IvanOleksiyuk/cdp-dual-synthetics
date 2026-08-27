import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import logging
import argparse
from  matplotlib.colors import LinearSegmentedColormap
import jsonpickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metric_data(pickle_files, reference_labels=None, dedicated=0, metric_type="auc"):
    """
    Load metric data from multiple pickle files into a structured format, optionally filtering by dedicated class.
    
    Args:
        pickle_files (list): List of paths to pickle files with metric results.
        reference_labels (list, optional): Labels for each reference. If None, filenames will be used.
        dedicated (int): If 1, only include results for the dedicated class of each reference.
        metric_type (str): Type of metric to extract ("auc", "mean_ref_vs_orig", "mean_ref_vs_fake", "std_ref_vs_orig", "std_ref_vs_fake").
        
    Returns:
        dict: Dictionary with metrics as keys and dataframes with references × dataset pairs as values.
    """
    # Check if we have valid files
    if not pickle_files:
        logger.error("No pickle files provided")
        return {}
    
    # Initialize data structure
    data = {}
    dataset_pairs = set()
    reference_labels = []

    # Load all pickle files
    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            result = pickle.load(f)
        print(f"Loaded pickle file: {pickle_file}")
        print(f"Result keys: {list(result.keys())}")
        # Extract metadata
        metadata = result.get("metadata", {})
        reference_label = metadata.get("reference_print_label", Path(pickle_file).stem)
        dedicated_class = metadata.get("dedicated_class", None)
        reference_labels.append(reference_label)
        
        # Extract results
        results = result.get("results", {})
        
        # Get all metrics and dataset pairs
        for pair, metrics_dict in results.items():
            # If dedicated=1, only include results for the dedicated class or if dedicated_class is None
            if dedicated == 1 and dedicated_class is not None and dedicated_class not in pair:
                continue
            
            dataset_pairs.add(pair)
            for metric_name, metric_data in metrics_dict.items():
                if metric_name not in data:
                    data[metric_name] = {}
                if reference_label not in data[metric_name]:
                    data[metric_name][reference_label] = {}
                # Extract the specified metric value from the metric data dictionary
                if isinstance(metric_data, dict):
                    metric_value = metric_data.get(metric_type, np.nan)
                else:
                    # Backward compatibility - assume it's AUC if not a dict
                    metric_value = metric_data if metric_type == "auc" else np.nan
                data[metric_name][reference_label][pair] = metric_value
        
        logger.info(f"Loaded data from {pickle_file} - Reference: {reference_label}, Dedicated Class: {dedicated_class}")
    
    # Verify loaded data
    logger.info(f"Total dataset pairs: {len(dataset_pairs)}")
    logger.info(f"Metrics to plot: {', '.join(data.keys())}")
    
    return data, sorted(list(dataset_pairs))

def plot_metric_tables(data, dataset_pairs, output_dir="plots/metric_comparison", cmap="viridis", figsize=(12, 10), baseline=None, value_display="diffabs", metric_type="auc"):
    """
    Create table plots for each metric showing specified metric values across references and dataset pairs.

    Note: This function only generates plots. Use `write_metric_summary_txt` to write the corresponding text summaries.

    Args:
        data (dict): Dictionary with metrics and their values organized by reference and dataset pair
        dataset_pairs (list): Ordered list of dataset pairs for consistent plotting
        output_dir (str): Directory to save the output plots
        cmap (str): Matplotlib colormap to use
        figsize (tuple): Figure size (width, height) in inches
        baseline (str, optional): Reference label to use as the baseline for comparison
        value_display (str, optional): How to display values in cells ("diff", "diffabs", "abs"). Default is "diffabs".
        metric_type (str): Type of metric being plotted (for labels and file naming)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a simpler dataset pair label for plotting (remove common prefixes)
    simplified_pairs = []
    for pair in dataset_pairs:
        parts = pair.split("_vs_")
        if len(parts) == 2:
            orig_label = parts[0]
            simplified_pairs.append(f"{orig_label[1:]}")  # Simplified label
        else:
            simplified_pairs.append(pair)

    # Plot for each metric
    for metric, references_data in data.items():
        references = list(references_data.keys())

        # Determine baseline index
        baseline_index = references.index(baseline) if baseline and baseline in references else None
        if baseline == "last":
            baseline_index = len(references) - 1
        if baseline is not None and baseline_index is None:
            logger.warning(f"Baseline reference '{baseline}' not found in data. Skipping baseline comparison.")
            logger.warning(f"Available references: {references}")
            baseline_index = None

        # Create array for heatmap
        metric_array = np.zeros((len(references), len(dataset_pairs)))
        diff_array = np.zeros_like(metric_array) if baseline_index is not None else None
        for i, ref in enumerate(references):
            for j, pair in enumerate(dataset_pairs):
                metric_value = references_data[ref].get(pair, np.nan)
                metric_array[i, j] = metric_value
                if baseline_index is not None:
                    baseline_value = references_data[references[baseline_index]].get(pair, np.nan)
                    diff_array[i, j] = metric_value - baseline_value if not np.isnan(baseline_value) else np.nan

        # Create the figure and heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Define colormap for differences if baseline is set
        if baseline_index is not None:
            valid_diffs = diff_array[~np.isnan(diff_array)]
            if len(valid_diffs) > 0:
                max_abs_diff = np.max(np.abs(valid_diffs))
                scale_range = max(max_abs_diff * 1.2, 0.01)
            else:
                scale_range = 0.2

            norm = mpl.colors.TwoSlopeNorm(vmin=-scale_range, vcenter=0, vmax=scale_range)
            c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
            v = [0, .15, .4, .5, 0.6, .9, 1.]
            l = list(zip(v, c))
            cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
            im = ax.imshow(diff_array, cmap=cmap, norm=norm)
            cbar_label = "Difference to Baseline"
        else:
            if metric_type == "auc":
                norm = mpl.colors.Normalize(vmin=0.5, vmax=1.0)
                cbar_label = "AUC Score"
            elif "mean" in metric_type:
                valid_values = metric_array[~np.isnan(metric_array)]
                if len(valid_values) > 0:
                    vmin, vmax = np.percentile(valid_values, [5, 95])
                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = mpl.colors.Normalize()
                cbar_label = metric_type.replace("_", " ").title()
            else:
                valid_values = metric_array[~np.isnan(metric_array)]
                if len(valid_values) > 0:
                    vmin, vmax = np.percentile(valid_values, [5, 95])
                    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = mpl.colors.Normalize()
                cbar_label = metric_type.replace("_", " ").title()
            im = ax.imshow(metric_array, cmap=cmap, norm=norm)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

        # Set ticks and labels
        ax.set_xticks(np.arange(len(simplified_pairs)))
        ax.set_yticks(np.arange(len(references)))
        ax.set_xticklabels(simplified_pairs, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(references)

        # Add grid lines
        ax.set_xticks(np.arange(-.5, len(simplified_pairs), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(references), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

        # Add title and labels
        metric_display_name = metric_type.replace("_", " ").title()
        ax.set_title(f"{metric_display_name} Comparison for {metric}")
        ax.set_ylabel("Reference Dataset")
        ax.set_xlabel("Dataset Pair (Original→Fake)")

        # Annotate each cell with its value
        for i in range(len(references)):
            for j, pair in enumerate(dataset_pairs):
                diff_value = diff_array[i, j] if baseline_index is not None else None
                abs_value = metric_array[i, j]
                if not np.isnan(abs_value):
                    if baseline_index is not None:
                        if value_display == "diff":
                            text = f"{diff_value:.3f}"
                        elif value_display == "diffabs":
                            text = f"{diff_value:.3f}\n[{abs_value:.3f}]"
                        elif value_display == "abs":
                            text = f"{abs_value:.3f}"
                        else:
                            text = f"{diff_value:.3f}\n[{abs_value:.3f}]"
                    else:
                        text = f"{abs_value:.3f}"

                    if baseline_index is None:
                        if metric_type == "auc":
                            text_color = "white" if abs_value < 0.75 else "black"
                        else:
                            valid_values = metric_array[~np.isnan(metric_array)]
                            threshold = np.median(valid_values) if len(valid_values) > 0 else 0
                            text_color = "white" if abs_value < threshold else "black"
                    else:
                        text_color = "black" if diff_value < 0 else "black"
                    ax.text(j, i, text, ha="center", va="center", color=text_color, fontweight="bold")

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, f"{metric_type}_comparison_{metric}.png")
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved {metric} {metric_type} comparison plot to {output_path}")
        plt.close()

def write_metric_summary_txt(data, dataset_pairs, output_dir="plots/metric_comparison", baseline=None, value_display="diffabs", metric_type="auc"):
    """
    Write a text summary file for each metric type, aggregating values across references and dataset pairs.

    Args:
        data (dict): Dictionary with metrics and their values organized by reference and dataset pair
        dataset_pairs (list): Ordered list of dataset pairs for consistent ordering in the table
        output_dir (str): Directory where the summary text file will be saved
        baseline (str, optional): Baseline reference label
        value_display (str, optional): How to display values ("diff", "diffabs", "abs")
        metric_type (str): Metric type name used for the file name and headers
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_lines = []

    for metric, references_data in data.items():
        references = list(references_data.keys())

        baseline_index = references.index(baseline) if baseline and baseline in references else None
        if baseline == "last":
            baseline_index = len(references) - 1
        if baseline is not None and baseline_index is None:
            logger.warning(f"Baseline reference '{baseline}' not found in data. Skipping baseline comparison.")
            logger.warning(f"Available references: {references}")
            baseline_index = None

        metric_array = np.zeros((len(references), len(dataset_pairs)))
        diff_array = np.zeros_like(metric_array) if baseline_index is not None else None
        for i, ref in enumerate(references):
            for j, pair in enumerate(dataset_pairs):
                metric_value = references_data[ref].get(pair, np.nan)
                metric_array[i, j] = metric_value
                if baseline_index is not None:
                    baseline_value = references_data[references[baseline_index]].get(pair, np.nan)
                    diff_array[i, j] = metric_value - baseline_value if not np.isnan(baseline_value) else np.nan

        header = ["Reference"] + [pair for pair in dataset_pairs]
        metric_display_name = metric_type.replace("_", " ").title()
        txt_lines.append(f"{metric_display_name} Comparison for {metric}")
        txt_lines.append("\t".join(header))
        for i, ref in enumerate(references):
            row = [ref]
            for j, pair in enumerate(dataset_pairs):
                abs_value = metric_array[i, j]
                if baseline_index is not None:
                    diff_value = diff_array[i, j]
                    if value_display == "diff":
                        cell = f"{diff_value:.3f}"
                    elif value_display == "diffabs":
                        cell = f"{diff_value:.3f}[{abs_value:.3f}]"
                    elif value_display == "abs":
                        cell = f"{abs_value:.3f}"
                    else:
                        cell = f"{diff_value:.3f}[{abs_value:.3f}]"
                else:
                    cell = f"{abs_value:.3f}"
                row.append(cell)
            txt_lines.append("\t".join(row))
        txt_lines.append("")

    txt_path = os.path.join(output_dir, f"{metric_type}_comparison_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))
    logger.info(f"Saved {metric_type} summary table to {txt_path}")

def write_metric_summary_json(data, dataset_pairs, output_dir="plots/metric_comparison", baseline=None, value_display="diffabs", metric_type="auc"):
    """
    Write a JSON summary (via jsonpickle) capturing the same table content as the TXT summary,
    with both human-readable cells and numeric arrays for easier re-loading.

    The JSON structure:
    {
      "metric_type": str,
      "dataset_pairs": list[str],
      "baseline": Optional[str],
      "tables": [
        {
          "metric": str,
          "references": list[str],
          "cells": list[list[str]],  # same formatting as TXT rows (excluding header)
          "abs_values": list[list[float]],
          "diff_values": Optional[list[list[float]]]
        }, ...
      ]
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "metric_type": metric_type,
        "dataset_pairs": dataset_pairs,
        "baseline": baseline,
        "tables": []
    }

    for metric, references_data in data.items():
        references = list(references_data.keys())

        baseline_index = references.index(baseline) if baseline and baseline in references else None
        if baseline == "last":
            baseline_index = len(references) - 1
        if baseline is not None and baseline_index is None:
            logger.warning(f"Baseline reference '{baseline}' not found in data. Skipping baseline comparison.")
            logger.warning(f"Available references: {references}")
            baseline_index = None

        metric_array = np.zeros((len(references), len(dataset_pairs)))
        diff_array = np.zeros_like(metric_array) if baseline_index is not None else None
        for i, ref in enumerate(references):
            for j, pair in enumerate(dataset_pairs):
                metric_value = references_data[ref].get(pair, np.nan)
                metric_array[i, j] = metric_value
                if baseline_index is not None:
                    baseline_value = references_data[references[baseline_index]].get(pair, np.nan)
                    diff_array[i, j] = metric_value - baseline_value if not np.isnan(baseline_value) else np.nan

        # Build cell strings like in TXT
        cells = []
        for i, ref in enumerate(references):
            row_cells = []
            for j, pair in enumerate(dataset_pairs):
                abs_value = metric_array[i, j]
                if baseline_index is not None:
                    diff_value = diff_array[i, j]
                    if value_display == "diff":
                        cell = f"{diff_value:.3f}"
                    elif value_display == "diffabs":
                        cell = f"{diff_value:.3f}[{abs_value:.3f}]"
                    elif value_display == "abs":
                        cell = f"{abs_value:.3f}"
                    else:
                        cell = f"{diff_value:.3f}[{abs_value:.3f}]"
                else:
                    cell = f"{abs_value:.3f}"
                row_cells.append(cell)
            cells.append(row_cells)

        summary["tables"].append({
            "metric": metric,
            "references": references,
            "cells": cells,
            "abs_values": metric_array.tolist(),
            "diff_values": diff_array.tolist() if diff_array is not None else None,
        })

    json_path = os.path.join(output_dir, f"{metric_type}_comparison_summary.json")
    # Configure pretty printing for the json backend
    try:
        jsonpickle.set_preferred_backend('json')
        jsonpickle.set_encoder_options('json', indent=2)
    except Exception:
        pass
    with open(json_path, "w") as f:
        f.write(jsonpickle.encode(summary, make_refs=False))
    logger.info(f"Saved {metric_type} JSON summary to {json_path}")

def generate_dataset_pairs_from_captures(captures_used):
    """
    Generate dataset pairs from captures_used list.
    
    Args:
        captures_used (list): List of capture names to generate pairs from.
        
    Returns:
        list: List of dataset pairs in format "o{capture}_vs_f{capture}"
    """
    dataset_pairs = []
    for capture in captures_used:
        pair = f"{capture}_vs_f{capture[1:]}"
        dataset_pairs.append(pair)
    return dataset_pairs

def compare_metrics_across_references(pickle_files, output_dir="plots/metric_comparison", cmap="viridis", baseline=None, value_display="diffabs", dedicated=0, metric_types=None, captures_used=None):
    """
    Create comparison plots showing metric performance across different references for each metric type.
    
    Args:
        pickle_files (list): List of paths to pickle files with metric results.
        output_dir (str): Directory to save the output plots.
        cmap (str): Matplotlib colormap to use for plots.
        baseline (str, optional): Reference label to use as the baseline for comparison.
        value_display (str, optional): How to display values in cells ("diff", "diffabs", "abs"). Default is "diffabs".
        dedicated (int): If 1, only include results for the dedicated class of each reference.
        metric_types (list, optional): List of metric types to plot. If None, defaults to ["auc"].
        captures_used (list, optional): List of captures to use for generating dataset pairs. If None, dataset pairs will be extracted from pickle files.
    """
    if metric_types is None:
        metric_types = ["auc"]
    
    # Load data from all pickle files
    reference_labels = []
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                result = pickle.load(f)
            
            # Extract reference label from the pickle file
            print(result["metadata"])
            reference_label = result["metadata"].get("reference_print_label", Path(pickle_file).stem)
            reference_labels.append(reference_label)
        except Exception as e:
            logger.error(f"Error loading pickle file {pickle_file}: {e}")
            continue
    
    # Handle "last" baseline case
    if baseline in ["none", "None", "null", "Null"]:
        baseline = None
    
    # Create plots for each metric type
    for metric_type in metric_types:
        logger.info(f"Processing {metric_type} metrics...")
        
        # Load data for this specific metric type
        data, dataset_pairs_from_data = load_metric_data(pickle_files, reference_labels, dedicated=dedicated, metric_type=metric_type)
        
        # Use command-line provided captures if available, otherwise use dataset pairs from data
        if captures_used is not None:
            dataset_pairs = generate_dataset_pairs_from_captures(captures_used)
            logger.info(f"Using command-line provided captures to generate dataset pairs: {dataset_pairs}")
        else:
            dataset_pairs = dataset_pairs_from_data
            logger.info(f"Using dataset pairs from loaded data: {dataset_pairs}")
        
        if not data:
            logger.warning(f"No valid data found for {metric_type}")
            continue
        
    # Generate plots comparing all references for this metric type
    plot_metric_tables(data, dataset_pairs, output_dir, cmap, baseline=baseline, value_display=value_display, metric_type=metric_type)
    # Write text summary for this metric type
    write_metric_summary_txt(data, dataset_pairs, output_dir, baseline=baseline, value_display=value_display, metric_type=metric_type)
    # Write JSON summary for this metric type
    write_metric_summary_json(data, dataset_pairs, output_dir, baseline=baseline, value_display=value_display, metric_type=metric_type)
        
    logger.info(f"Completed {metric_type} comparison plots for {len(data)} metrics")

if __name__ == "__main__":
    # Set up argument Parser
    parser = argparse.ArgumentParser(description="Generate metric comparison plots from pickle files.")
    parser.add_argument("--input_pickles", nargs='+', required=True, help="List of input pickle files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap to use for plots.")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline model for comparison (use 'last' for the last model).")
    parser.add_argument("--value_display", type=str, default="diffabs", choices=["diff", "diffabs", "abs"], help="How to display values in cells ('diff', 'diffabs', 'abs'). Default is 'diffabs'.")
    parser.add_argument("--dedicated", type=int, default=0, choices=[0, 1], help="Filter results by dedicated class (1) or include all (0). Default is 0.")
    parser.add_argument("--metric_types", nargs='+', default=["auc"], 
                        choices=["auc", "mean_ref_vs_orig", "mean_ref_vs_fake", "std_ref_vs_orig", "std_ref_vs_fake"],
                        help="List of metric types to plot. Default is ['auc'].")
    parser.add_argument("--captures_used", nargs='+', default=None, 
                        help="List of captures to use for generating dataset pairs. If not provided, dataset pairs will be extracted from pickle files.")
    
    args = parser.parse_args()
    
    # Run the comparison function with the provided arguments
    compare_metrics_across_references(
        pickle_files=args.input_pickles,
        output_dir=args.output_dir,
        cmap=args.cmap,
        baseline=args.baseline,
        value_display=args.value_display,
        dedicated=args.dedicated,
        metric_types=args.metric_types,
        captures_used=args.captures_used
    )
