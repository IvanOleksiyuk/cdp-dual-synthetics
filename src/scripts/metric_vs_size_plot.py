import os
import re
import logging
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import jsonpickle
import importlib

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def improve_label(label, to_class=""):
    label_map={"tem": "$t$",
               "o55EpsonDUP": "$x_{Epson}$",
               "o55iP15mDUP": "$x_{iP15m}$",
               "o55iP15wDUP": "$x_{iP15w}$",
               "o55iP14wDUP": "$x_{iP14w}$",
               "o55iP14mDUP": "$x_{iP14m}$",
               "o55iPXSoDUP": "$x_{iPXSo}$",
               "o55iP12wDUP": "$x_{iP12w}$",
               "l2cond_EpsoniP15miP14wiPXSo": "$\hat{x}_{"+f"{to_class}"+",L_2}$[8]",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55Epson": "$\hat{x}_{Epson \\rightarrow "+f"{to_class}"+"}$",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iP15m": "$\hat{x}_{iP15m \\rightarrow "+f"{to_class}"+"}$",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iP14w": "$\hat{x}_{iP14w \\rightarrow "+f"{to_class}"+"}$",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iPXSo": "$\hat{x}_{iPXSo \\rightarrow "+f"{to_class}"+"}$",
               "o76EpsonDUP": "$x_{Epson}$",
               "o76iP15mDUP": "$x_{iP15m}$",
               "o76iP14wDUP": "$x_{iP14w}$",
               "o76iPXSoDUP": "$x_{iPXSo}$",
               "roman": "$\hat{x}_{"+f"{to_class}"+",L_{comb}}$[8]",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o76Epson": "$\hat{x}_{Epson \\rightarrow "+f"{to_class}"+"}$",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o76iP15m": "$\hat{x}_{iP15m \\rightarrow "+f"{to_class}"+"}$",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o76iP14w": "$\hat{x}_{iP14w \\rightarrow "+f"{to_class}"+"}$",
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o76iPXSo": "$\hat{x}_{iPXSo \\rightarrow "+f"{to_class}"+"}$"
               }
    if label in label_map:
        return label_map[label]
    if label.startswith("l2condtransport_maxinfo_MR50"):
        from_class = label.split("_from_")[-1].replace("o55","").replace("o76","")
        return "$\hat{x}_{"+from_class+" \\rightarrow "+f"{to_class}"+"}$"
    return label

def special_colors_n_styles(label):
    style_map={"tem": {"color": "black"},
               "roman": {"color": "green", "linestyle": '-.'},
               "o55EpsonDUP": {"color": "dodgerblue"},
               "o55iP15mDUP": {"color": "blue"},
               "o55iP15wDUP": {"color": "darkblue"},
               "o55iP14mDUP": {"color": "darkturquoise"},
               "o55iP14wDUP": {"color": "cadetblue"},
               "o55iPXSoDUP": {"color": "lightskyblue"},
               "o55iP12wDUP": {"color": "springgreen"},
               "l2cond_EpsoniP15miP14wiPXSo": {"color": "green", "linestyle":":"},
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iP15m": {"color":"tomato", "linestyle":"--"},
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iPXSo": {"color":"tomato", "linestyle":"--"},
               "o76EpsonDUP": {"color": "skyblue"},
               "o76iP15mDUP": {"color": "blue"},
               "o76iP15wDUP": {"color": "darkblue"},
               "o76iP14mDUP": {"color": "darkturquoise"},
               "o76iP14wDUP": {"color": "paleturquoise"},
               "l2cond_EpsoniP15miP14wiPXSo": {"color": "green", "linestyle":":"},
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o76iP15m": {"color":"tomato", "linestyle":"--"},
               "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o76iPXSo": {"color": "tomato", "linestyle":"--"}}
    if label in style_map:
        return style_map[label]
    if label.startswith("l2condtransport_maxinfo_MR50"):
        from_class = label.split("_from_")[-1].replace("o55","").replace("o76","")
        if from_class=="Epson":
            return {"color":"brown", "linestyle":"--", "alpha":0.8}
        if from_class=="iP15m":
            return {"color":"red", "linestyle":"--", "alpha":0.8}
        if from_class=="iP15w":
            return {"color":"maroon", "linestyle":"--", "alpha":0.8}
        if from_class=="iP14m":
            return {"color":"fuchsia", "linestyle":"--", "alpha":0.8}
        if from_class=="iP14w":
            return {"color":"deeppink", "linestyle":"--", "alpha":0.8}
        if from_class=="iP12w":
            return {"color":"chocolate", "linestyle":"--", "alpha":0.8}
        if from_class=="iPXSo":
            return {"color":"tomato", "linestyle":"--", "alpha":0.8}
        return {"color":"tomato", "linestyle":"--", "alpha":0.8}
    return None

def improve_title(title: str) -> str:
    """Improve the plot title by making it more descriptive."""
    title_map = {
        "o55iPXSo_vs_f55iPXSo": "iPhone XS original vs fake authentication",
        "o55iP15m_vs_f55iP15m": "iPhone 15 macro original vs fake authentication",
        "o55iP15w_vs_f55iP15w": "iPhone 15 wide original vs fake authentication",
        "o55iP14m_vs_f55iP14m": "iPhone 14 macro original vs fake authentication",
        "o55iP14w_vs_f55iP14w": "iPhone 14 wide original vs fake authentication",
        "o55Epson_vs_f55Epson": "Epson original vs fake authentication",
        "o55iP12w_vs_f55iP12w": "iPhone 12 original vs fake authentication",

        "o76iPXSo_vs_f76iPXSo": "iPhone XS original vs fake authentication",
        "o76iP15m_vs_f76iP15m": "iPhone 15 macro original vs fake authentication",
    }
    return title_map.get(title, title) #

#
# Note: previously there was an unfinished helper here; removing to avoid syntax errors.

# Build mapping from subfolder naming scheme to numeric block sizes
def build_size_map(block_size: int, block_stride: int, full_image_size: int) -> Dict[str, int]:
    # number of strides that still fit within the image (non-negative)
    if block_stride <= 0:
        raise ValueError("block_stride must be > 0")
    steps = max(0, (full_image_size - block_size) // block_stride)
    size_map: Dict[str, int] = {
        "blocks_separately": int(block_size),
        "average_blocks": int(block_size + steps * block_stride),
    }
    for i in range(1, 20):
        size_map[f"blocks_combine-{i}"] = int(block_size + block_stride * (i - 1))
    return size_map

def load_summary_json(json_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON summary file produced by write_metric_summary_json.

    Returns a dict or None on failure.
    """
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            data = jsonpickle.decode(f.read())
        return data
    except Exception as e:
        logger.warning(f"Failed to load JSON '{json_path}': {e}")
        return None


def extract_abs_value(summary: Dict[str, Any], metric: str, reference: str, dataset_pair: str) -> Optional[float]:
    """Extract absolute metric value for a given metric, reference and dataset_pair from a summary dict.

    summary structure:
    {
      "metric_type": str,
      "dataset_pairs": [str],
      "baseline": Optional[str],
      "tables": [
         {
           "metric": str,
           "references": [str],
           "cells": [[str]],
           "abs_values": [[float]],
           "diff_values": [[float]] | None
         }, ...
      ]
    }
    """
    tables = summary.get("tables", [])
    table = None
    for t in tables:
        if t.get("metric") == metric:
            table = t
            break
    if table is None:
        logger.debug(f"Metric '{metric}' not found in summary tables.")
        return None

    references = table.get("references", [])
    dataset_pairs = summary.get("dataset_pairs", [])

    try:
        i = references.index(reference)
    except ValueError:
        logger.debug(f"Reference '{reference}' not in references: {references}")
        return None

    try:
        j = dataset_pairs.index(dataset_pair)
    except ValueError:
        logger.debug(f"Dataset pair '{dataset_pair}' not in dataset_pairs: {dataset_pairs}")
        return None

    abs_values = table.get("abs_values")
    if abs_values is None:
        return None

    try:
        val = float(abs_values[i][j])
        if np.isnan(val):
            return None
        return val
    except Exception:
        return None


def collect_values_vs_subfolder(
    root_dir: str,
    metric_type: str,
    metric: str,
    reference: str,
    dataset_pair: str,
    auc_log1m: bool = False,
    auc_log1m_upper: float = 0.999,
    json_filename: Optional[str] = None,
) -> List[Tuple[str, Optional[float], Optional[bool]]]:
    """Scan immediate subfolders of root_dir, load JSON summaries, and collect metric values.

    If json_filename is None, uses f"{metric_type}_comparison_summary.json".
    Returns a list of (subfolder_name, value or None if missing).
    """
    if json_filename is None:
        json_filename = f"{metric_type}_comparison_summary.json"

    results: List[Tuple[str, Optional[float], Optional[bool]]] = []
    if not os.path.isdir(root_dir):
        logger.error(f"Root directory does not exist: {root_dir}")
        return results

    for name in sorted(os.listdir(root_dir)):
        subpath = os.path.join(root_dir, name)
        if not os.path.isdir(subpath):
            continue
        if name == "final_plots":
            continue
        json_path = os.path.join(subpath, json_filename)
        summary = load_summary_json(json_path)
        if summary is None:
            logger.info(f"Skipping '{name}': missing {json_filename}")
            results.append((name, None, None))
            continue
        if summary.get("metric_type") != metric_type:
            logger.info(f"Skipping '{name}': metric_type mismatch ({summary.get('metric_type')} != {metric_type})")
            results.append((name, None, None))
            continue
        val = extract_abs_value(summary, metric, reference, dataset_pair)
        clipped_flag: Optional[bool] = None
        # If metric is AUC-like and below 0.5, flip to 1 - AUC
        if val is not None:
            mt_lower = str(metric_type).lower()
            if "auc" in mt_lower and val < 0.5:
                logger.debug(f"Flipping AUC value {val:.6f} -> {1.0 - val:.6f} for '{name}'")
                val = 1.0 - val
            # Optional transform when flag active: clip then complement AUC
            if "auc" in mt_lower and auc_log1m:
                # Clip upper to avoid zero after complement (ensures positive for log-scale plotting)
                upper = float(auc_log1m_upper)
                if not np.isfinite(upper) or upper <= 0 or upper >= 1:
                    upper = 0.999
                if val > upper:
                    val = upper
                    clipped_flag = True
                else:
                    clipped_flag = False
                val = 1.0 - val
        if val is None:
            logger.info(f"Value not found for metric='{metric}', reference='{reference}', pair='{dataset_pair}' in '{name}'")
        results.append((name, val, clipped_flag))

    return results


def natural_key(text: str) -> List[Any]:
    """Key for natural sorting (numbers within strings sorted numerically)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def combine_transport_series(series_by_ref: Dict[str, List[Tuple[int, Optional[float], str, Optional[bool]]]]):
    """Combine all transport references (prefix match) into mean/min/max per block size.

    Returns a dict with keys: sizes, mean, min, max, label. Returns None if no transport refs.
    """
    prefix = "l2condtransport_maxinfo_MR50"
    transport_only = {k: v for k, v in series_by_ref.items() if k.startswith(prefix)}
    if not transport_only:
        return None

    size_to_vals: Dict[int, List[float]] = {}
    for _, points in transport_only.items():
        for bs, val, *_ in points:
            if val is None:
                continue
            size_to_vals.setdefault(bs, []).append(float(val))

    if not size_to_vals:
        return None

    sizes = sorted(size_to_vals.keys())
    means: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    for bs in sizes:
        vals = size_to_vals[bs]
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))

    return {
        "sizes": sizes,
        "mean": means,
        "min": mins,
        "max": maxs,
    }


def combine_other_camera_series(
    series_by_ref: Dict[str, List[Tuple[int, Optional[float], str, Optional[bool]]]],
    to_class: str,
):
    """Combine all references ending with 'DUP' except the one matching the target class.

    Returns a tuple of (band_dict, combined_keys). band_dict mirrors combine_transport_series output
    with an added 'label'. combined_keys lists references that were combined and should be removed
    from individual plotting.
    """
    eligible_keys = []
    target = str(to_class) if to_class is not None else ""
    for ref in series_by_ref.keys():
        if not ref.endswith("DUP"):
            continue
        if target and target in ref:
            continue
        eligible_keys.append(ref)

    if not eligible_keys:
        return None, set()

    size_to_vals: Dict[int, List[float]] = {}
    for ref in eligible_keys:
        for bs, val, *_ in series_by_ref.get(ref, []):
            if val is None:
                continue
            size_to_vals.setdefault(bs, []).append(float(val))

    if not size_to_vals:
        return None, set()

    sizes = sorted(size_to_vals.keys())
    means: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []
    for bs in sizes:
        vals = size_to_vals[bs]
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))

    band = {
        "sizes": sizes,
        "mean": means,
        "min": mins,
        "max": maxs,
    }
    return band, set(eligible_keys)


def get_references_from_any_json(
    root_dir: str,
    metric_type: str,
    metric: str,
    json_filename: Optional[str] = None,
) -> List[str]:
    """Find the first available JSON and extract the list of references for the given metric.

    Returns an empty list if none found.
    """
    if json_filename is None:
        json_filename = f"{metric_type}_comparison_summary.json"

    if not os.path.isdir(root_dir):
        return []

    for name in sorted(os.listdir(root_dir)):
        subpath = os.path.join(root_dir, name)
        if not os.path.isdir(subpath):
            continue
        json_path = os.path.join(subpath, json_filename)
        summary = load_summary_json(json_path)
        print(summary)
        if summary is None or summary.get("metric_type") != metric_type:
            continue
        # Find table for metric
        for t in summary.get("tables", []):
            if t.get("metric") == metric:
                refs = t.get("references", [])
                return refs if isinstance(refs, list) else []
    return []


def plot_vs_blocksize(
    points: List[Tuple[int, Optional[float], str, Optional[bool]]],
    title: str,
    ylabel: str,
    output_path: str,
    style: str = "scatter",
    annotate: bool = True,
    reference_label: Optional[str] = None,
    auto_ylim: bool = False,
    log_y: bool = False,
    truey: bool = False,
    do_legend_outside_plot: int = 0,
):
    """Plot metric value versus numeric block size.

    points: list of (block_size, value, label)
    """
    if not points:
        logger.warning("No data points to plot.")
        return

    sizes = [p[0] for p in points]
    values = [p[1] for p in points]
    labels = [p[2] for p in points]
    clipped_flags = [bool(p[3]) if len(p) > 3 and p[3] is not None else False for p in points]
    x_ticks_all = sorted(set(sizes))
    if len(x_ticks_all) % 2 == 0:
        # Keep first and last; if adjacency is unavoidable, keep it at the beginning.
        x_ticks = [x_ticks_all[0]] + x_ticks_all[1::2]
    else:
        x_ticks = x_ticks_all[::2]

    if do_legend_outside_plot > 0:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig, ax = plt.subplots(figsize=(8, 5))

    valid_idx = [i for i, v in enumerate(values) if v is not None]

    color_kwargs = {}
    if reference_label:
        spec = special_colors_n_styles(reference_label)
        if spec:
            if isinstance(spec, dict):
                color_kwargs.update(spec)
            else:  # string color
                color_kwargs['color'] = spec

    # Choose marker for clipped points depending on axis interpretation
    clipped_marker = '^' if truey else 'v'

    if style == "line":
        y = [values[i] if i in valid_idx else np.nan for i in range(len(values))]
        # draw line without markers
        ax.plot(sizes, y, **{k:v for k,v in color_kwargs.items() if k in ("color","linestyle","linewidth","alpha")})
        # overlay markers: unclipped circles, clipped downward triangles
        unclipped_idx = [i for i in valid_idx if not clipped_flags[i]]
        clipped_idx = [i for i in valid_idx if clipped_flags[i]]
        if unclipped_idx:
            ax.scatter([sizes[i] for i in unclipped_idx], [values[i] for i in unclipped_idx], s=20, marker='o', **{k:v for k,v in color_kwargs.items() if k=="color"})
        if clipped_idx:
            ax.scatter([sizes[i] for i in clipped_idx], [values[i] for i in clipped_idx], s=20, marker=clipped_marker, **{k:v for k,v in color_kwargs.items() if k=="color"})
    else:
        # scatter only: split by clipped flag
        unclipped_idx = [i for i in valid_idx if not clipped_flags[i]]
        clipped_idx = [i for i in valid_idx if clipped_flags[i]]
        if unclipped_idx:
            ax.scatter([sizes[i] for i in unclipped_idx], [values[i] for i in unclipped_idx], s=20, marker='o', **{k:v for k,v in color_kwargs.items() if k=="color"})
        if clipped_idx:
            ax.scatter([sizes[i] for i in clipped_idx], [values[i] for i in clipped_idx], s=20, marker=clipped_marker, **{k:v for k,v in color_kwargs.items() if k=="color"})

    ax.set_xlabel("Window size (px)")
    ax.set_xticks(x_ticks)
    ax.set_title(improve_title(title))
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)

    if log_y:
        # Use logarithmic y-scale; avoid auto y-limits tuned for linear scale
        ax.set_yscale("log")
        # When using the AUC transform mode, enforce fixed limits
        ax.set_ylim(0.0008, 0.5)
        # If requested, invert axis and relabel ticks to show true values (e.g., AUC instead of 1-AUC)
        if truey:
            ax.invert_yaxis()
            try:
                _mt = importlib.import_module('matplotlib.ticker')
                _FuncFormatter = getattr(_mt, 'FuncFormatter', None)
                if _FuncFormatter is not None:
                    ax.yaxis.set_major_formatter(_FuncFormatter(lambda t, pos: f"{max(0.0, min(1.0, 1.0 - float(t))):.3f}"))
                else:
                    raise ImportError('FuncFormatter not available')
            except Exception:
                # Fallback: set static tick labels for current ticks (may desync on pan/zoom)
                ticks = ax.get_yticks()
                ax.set_yticklabels([f"{max(0.0, min(1.0, 1.0 - float(t))):.3f}" for t in ticks])

    if annotate:
        for i in valid_idx:
            ax.annotate(f"{values[i]:.3f}", (sizes[i], values[i]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    if (not log_y) and auto_ylim:
        # Dynamic y-limit rule (single curve fallback): lower bound = max(min(y), 0.8); upper fixed 1.05
        valid_values = [values[i] for i in valid_idx]
        if valid_values:
            lower_bound = min(valid_values)
            if lower_bound < 0.8:
                lower_bound = 0.8
        else:
            lower_bound = 0.8
        ax.set_ylim(lower_bound, 1.05)

    if do_legend_outside_plot > 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        plt.tight_layout(rect=[0,0.08,1,1])
    else:
        plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def plot_vs_blocksize_multi(
    series_by_ref: Dict[str, List[Tuple[int, Optional[float], str, Optional[bool]]]],
    title: str,
    ylabel: str,
    output_path: str,
    style: str = "line",
    annotate: bool = False,
    to_class: str = "",
    auto_ylim: bool = False,
    log_y: bool = False,
    truey: bool = False,
    do_legend_outside_plot: int = 0,
    transport_band: Optional[Dict[str, List[float]]] = None,
    other_camera_band: Optional[Dict[str, List[float]]] = None,
):
    """Plot multiple curves, one per reference, metric vs block size.

    series_by_ref maps reference -> list of (block_size, value, subfolder_label), expected sorted by block_size.
    """
    if not series_by_ref and transport_band is None and other_camera_band is None:
        logger.warning("No series to plot.")
        return

    if do_legend_outside_plot > 0:
        fig, ax = plt.subplots(figsize=(5, 6))
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
    all_x_ticks = set()

    if other_camera_band is not None:
        sizes = other_camera_band.get("sizes", [])
        all_x_ticks.update(sizes)
        means = other_camera_band.get("mean", [])
        mins = other_camera_band.get("min", [])
        maxs = other_camera_band.get("max", [])
        label = other_camera_band.get("label", "$x_{C2}\, C_2 \\neq C_1$ (combined)")
        color = "steelblue"
        if sizes and means:
            ax.plot(sizes, means, label=label, color=color, linewidth=2.0)
            if mins and maxs:
                ax.fill_between(sizes, mins, maxs, color=color, alpha=0.12, label=f"{label} min-max")
        else:
            logger.warning("Other-camera combination requested but no data available.")

    if transport_band is not None: #
        sizes = transport_band.get("sizes", [])
        all_x_ticks.update(sizes)
        means = transport_band.get("mean", [])
        mins = transport_band.get("min", [])
        maxs = transport_band.get("max", [])
        label = transport_band.get("label", "$\hat{x}_{C_2 \\rightarrow "+f"{to_class}"+"}$ (combined)")
        color = "tomato"
        if sizes and means:
            ax.plot(sizes, means, label=label, color=color, linewidth=2.0)
            if mins and maxs:
                ax.fill_between(sizes, mins, maxs, color=color, alpha=0.15, label=f"{label} min-max")
        else:
            logger.warning("Transport combination requested but no data available.")

    for ref, points in series_by_ref.items():
        if not points:
            continue
        sizes = [p[0] for p in points]
        all_x_ticks.update(sizes)
        values = [p[1] for p in points]
        clipped_flags = [bool(p[3]) if len(p) > 3 and p[3] is not None else False for p in points]
        valid_idx = [i for i, v in enumerate(values) if v is not None]
        if not valid_idx:
            logger.info(f"No valid values for reference '{ref}', skipping.")
            continue
        x = [sizes[i] for i in valid_idx]
        y = [values[i] for i in valid_idx]
        spec = special_colors_n_styles(ref)
        color_kwargs = {}
        if spec:
            if isinstance(spec, dict):
                color_kwargs.update(spec)
            else:
                color_kwargs['color'] = spec
        label_fmt = improve_label(ref, to_class=to_class)
        if style == "scatter":
            # Split into unclipped and clipped; provide label only once to keep legend clean
            unclipped_idx = [i for i in valid_idx if not clipped_flags[i]]
            clipped_idx = [i for i in valid_idx if clipped_flags[i]]
            label_for_unclipped = label_fmt if unclipped_idx or not clipped_idx else None
            if unclipped_idx:
                ax.scatter([sizes[i] for i in unclipped_idx], [values[i] for i in unclipped_idx], s=18, marker='o', label=label_for_unclipped, **{k:v for k,v in color_kwargs.items() if k=="color"})
            if clipped_idx:
                # Only set label if there was no unclipped points to ensure legend entry exists
                label_for_clipped = label_fmt if not unclipped_idx else None
                clipped_marker = '^' if truey else 'v'
                ax.scatter([sizes[i] for i in clipped_idx], [values[i] for i in clipped_idx], s=18, marker=clipped_marker, label=label_for_clipped, **{k:v for k,v in color_kwargs.items() if k=="color"})
        else:
            # Draw line without markers for legend, then overlay markers by clipped status without labels
            ax.plot(x, y, label=label_fmt, **{k:v for k,v in color_kwargs.items() if k in ("color","linestyle","linewidth","alpha")})
            unclipped_idx = [i for i in valid_idx if not clipped_flags[i]]
            clipped_idx = [i for i in valid_idx if clipped_flags[i]]
            if unclipped_idx:
                ax.scatter([sizes[i] for i in unclipped_idx], [values[i] for i in unclipped_idx], s=18, marker='o', **{k:v for k,v in color_kwargs.items() if k=="color"})
            if clipped_idx:
                clipped_marker = '^' if truey else 'v'
                ax.scatter([sizes[i] for i in clipped_idx], [values[i] for i in clipped_idx], s=18, marker=clipped_marker, **{k:v for k,v in color_kwargs.items() if k=="color"})
        if annotate:
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    ax.set_xlabel("Window size (px)")
    if all_x_ticks:
        x_ticks_all = sorted(all_x_ticks)
        if len(x_ticks_all) % 2 == 0:
            # Keep first and last; if adjacency is unavoidable, keep it at the beginning.
            x_ticks = [x_ticks_all[0]] + x_ticks_all[1::2]
        else:
            x_ticks = x_ticks_all[::2]
        ax.set_xticks(x_ticks)
    ax.set_title(improve_title(title))
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    if do_legend_outside_plot > 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        plt.tight_layout(rect=[0,0.08,1,1])
    else:
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
    
    if log_y:
        ax.set_yscale("log")
        # When using the AUC transform mode, enforce fixed limits
        ax.set_ylim(0.0008, 0.5)
        if truey:
            ax.invert_yaxis()
            try:
                _mt = importlib.import_module('matplotlib.ticker')
                _FuncFormatter = getattr(_mt, 'FuncFormatter', None)
                if _FuncFormatter is not None:
                    ax.yaxis.set_major_formatter(_FuncFormatter(lambda t, pos: f"{max(0.0, min(1.0, 1.0 - float(t))):.3f}"))
                else:
                    raise ImportError('FuncFormatter not available')
            except Exception:
                ticks = ax.get_yticks()
                ax.set_yticklabels([f"{max(0.0, min(1.0, 1.0 - float(t))):.3f}" for t in ticks])

    if (not log_y) and auto_ylim:
        # Compute dynamic y-limits across all curves per specification:
        # For each x, discard the lowest y (only if there are >=2 y values at that x); then take the lowest of remaining; lower bound = max(that, 0.8); upper = 1.05
        if transport_band is not None or other_camera_band is not None:
            values_pool = []
            if other_camera_band is not None:
                values_pool.extend(other_camera_band.get("min", []))
                values_pool.extend(other_camera_band.get("mean", []))
                values_pool.extend(other_camera_band.get("max", []))
            if transport_band is not None:
                values_pool.extend(transport_band.get("min", []))
                values_pool.extend(transport_band.get("mean", []))
                values_pool.extend(transport_band.get("max", []))
            valid_values = [v for v in values_pool if v is not None]
        else:
            x_to_vals: Dict[int, List[float]] = {}
            for ref, points in series_by_ref.items():
                for bs, v, _ in points:
                    if v is None:
                        continue
                    x_to_vals.setdefault(bs, []).append(v)
            remaining_vals: List[float] = []
            for bs, vals in x_to_vals.items():
                if not vals:
                    continue
                if len(vals) >= 2:
                    # discard the minimum
                    min_val = min(vals)
                    filtered = [vv for vv in vals if vv != min_val]
                    if not filtered:  # all identical case; keep one
                        filtered = vals
                    remaining_vals.extend(filtered)
                else:
                    remaining_vals.extend(vals)
            valid_values = remaining_vals

        if valid_values:
            lower_bound = min(valid_values)
            if lower_bound < 0.8:
                lower_bound = 0.8
        else:
            lower_bound = 0.8
        ax.set_ylim(lower_bound, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot metric absolute value vs subfolder name from JSON summaries.")
    parser.add_argument("--root_dir", required=True, help="Root folder containing subfolders with JSON summaries.")
    parser.add_argument("--metric_type", required=True, help="Metric type (e.g., 'auc').")
    parser.add_argument("--metric", required=True, help="Metric name inside the JSON 'tables' (e.g., a method/metric key).")
    parser.add_argument("--reference", required=True, help="Reference label to pick from JSON, or 'all' to plot a curve per reference.")
    parser.add_argument("--dataset_pair", required=True, help="Dataset pair string to pick from JSON.")
    parser.add_argument("--output_dir", default="plots/metric_vs_size", help="Directory to save the output plot.")
    parser.add_argument("--filename", default=None, help="Optional custom output filename.")
    parser.add_argument("--style", choices=["scatter", "line"], default="line", help="Plot style.")
    parser.add_argument("--sort", choices=["alpha", "natural", "none"], default="natural", help="Sort subfolder names.")
    parser.add_argument("--annotate", action="store_true", help="Annotate points with numeric values.")
    parser.add_argument("--auto_ylim", action="store_true", help="Enable dynamic y-axis limits rule (off by default).")
    # Optional AUC transform and plotting behavior
    parser.add_argument("--auc_log1m", action="store_true", help="If set, clip AUC (<= upper) and use 1 - AUC; plot with logarithmic y-scale.")
    parser.add_argument("--auc_log1m_upper", type=float, default=0.999, help="Upper limit to clip AUC before complement when --auc_log1m is set (default: 0.999).")
    parser.add_argument("--truey", action="store_true", help="Invert Y-axis and relabel ticks to show true values (e.g., AUC instead of 1-AUC).")
    # Block geometry parameters
    parser.add_argument("--block_size", type=int, default=128, help="Block size in pixels (default: 128).")
    parser.add_argument("--block_stride", type=int, default=64, help="Block stride in pixels (default: 64).")
    parser.add_argument("--full_image_size", type=int, default=684, help="Full image size in pixels for one dimension (default: 684).")
    parser.add_argument("--do_legend_outside_plot", type=int, default=0, help="If >0, place legend below the plot instead of on the plot itself.")
    parser.add_argument("--do_combine_transport", type=int, default=0, help="If >0, combine transport references into a mean curve with min/max band.")
    parser.add_argument("--do_combine_other_camera", type=int, default=0, help="If >0, combine non-target DUP references into a mean curve with min/max band.")

    args = parser.parse_args()

    # Build size map from provided geometry parameters
    try:
        size_map = build_size_map(args.block_size, args.block_stride, args.full_image_size)
    except Exception as e:
        logger.error(f"Invalid block parameters: {e}")
        return

    to_class_target = ""
    try:
        to_class_target = args.dataset_pair.split("_")[0][3:]
    except Exception:
        to_class_target = ""

    if args.reference.lower() == "all":
        # Gather references from any JSON file and plot a curve per reference
        refs = get_references_from_any_json(
            root_dir=args.root_dir,
            metric_type=args.metric_type,
            metric=args.metric,
        )
        if not refs:
            logger.error("Could not discover references from JSON summaries.")
            return
        print(f"Discovered references: {refs}")

        series_by_ref: Dict[str, List[Tuple[int, Optional[float], str, Optional[bool]]]] = {}
        for ref in refs:
            data = collect_values_vs_subfolder(
                root_dir=args.root_dir,
                metric_type=args.metric_type,
                metric=args.metric,
                reference=ref,
                dataset_pair=args.dataset_pair,
                auc_log1m=args.auc_log1m,
                auc_log1m_upper=args.auc_log1m_upper,
            )
            points: List[Tuple[int, Optional[float], str, Optional[bool]]] = []
            for name, val, clipped in data:
                size = size_map.get(name)
                if size is None:
                    logger.info(f"Unknown subfolder name for size_map, skipping: {name}")
                    continue
                points.append((int(size), val, name, clipped))
            points.sort(key=lambda t: t[0])
            series_by_ref[ref] = points

        transport_band = None
        if args.do_combine_transport:
            transport_band = combine_transport_series(series_by_ref)
            if transport_band is not None:
                # Remove transport entries from individual curves; keep others as usual
                series_by_ref = {k: v for k, v in series_by_ref.items() if not k.startswith("l2condtransport_maxinfo_MR50")}

        other_camera_band = None
        if args.do_combine_other_camera:
            other_camera_band, combined_keys = combine_other_camera_series(series_by_ref, to_class=to_class_target)
            if other_camera_band is not None:
                series_by_ref = {k: v for k, v in series_by_ref.items() if k not in combined_keys}

        title = f"{args.dataset_pair}"
        safe_pair = args.dataset_pair.replace("/", "-")
        fname = args.filename or f"{args.metric_type}_{args.metric}_ALL_{safe_pair}_vs_blocksize.png"
        output_path = os.path.join(args.output_dir, fname)

        # Y-axis label adjusted if AUC transform/log flag is active
        if "auc" in str(args.metric_type).lower():
            if args.auc_log1m:
                ylabel_text = (f"AUC for $f_{{{args.metric}}}$" if args.truey else f"1 − AUC for $f_{{{args.metric}}}$")
            else:
                ylabel_text = f"AUC for $f_{{{args.metric}}}$"
        else:
            ylabel_text = f"{args.metric_type.upper()} for $f_{{{args.metric}}}$"

        plot_vs_blocksize_multi(
            series_by_ref=series_by_ref,
            title=title,
            to_class=to_class_target,
            ylabel=ylabel_text,
            output_path=output_path,
            style=args.style,
            annotate=args.annotate,
            auto_ylim=args.auto_ylim,
            log_y=bool(args.auc_log1m),
            truey=bool(args.truey),
            do_legend_outside_plot=args.do_legend_outside_plot,
            transport_band=transport_band,
            other_camera_band=other_camera_band,
        )
    else:
        data = collect_values_vs_subfolder(
            root_dir=args.root_dir,
            metric_type=args.metric_type,
            metric=args.metric,
            reference=args.reference,
            dataset_pair=args.dataset_pair,
            auc_log1m=args.auc_log1m,
            auc_log1m_upper=args.auc_log1m_upper,
        )

        # Map subfolder names to numeric block sizes using size_map
        points: List[Tuple[int, Optional[float], str, Optional[bool]]] = []
        for name, val, clipped in data:
            size = size_map.get(name)
            if size is None:
                logger.info(f"Unknown subfolder name for size_map, skipping: {name}")
                continue
            points.append((int(size), val, name, clipped))

        # Sort by block size ascending
        points.sort(key=lambda t: t[0])

        title = f"{args.metric_type} [{args.metric}] @ {args.reference} / {args.dataset_pair}"
        if args.filename:
            fname = args.filename
        else:
            safe_pair = args.dataset_pair.replace("/", "-")
            fname = f"{args.metric_type}_{args.metric}_{args.reference}_{safe_pair}_vs_blocksize.png"
        output_path = os.path.join(args.output_dir, fname)

        # Y-axis label adjusted if AUC transform/log flag is active 
        if "auc" in str(args.metric_type).lower():
            if args.auc_log1m:
                ylabel_text = "AUC" if args.truey else "1 − AUC (log y)"
            else:
                ylabel_text = "AUC"
        else:
            ylabel_text = f"{args.metric_type} (abs)"

        plot_vs_blocksize(
            points=points,
            title=title,
            ylabel=ylabel_text,
            output_path=output_path,
            style=args.style,
            annotate=args.annotate,
            reference_label=args.reference,
            auto_ylim=args.auto_ylim,
            log_y=bool(args.auc_log1m),
            truey=bool(args.truey),
            do_legend_outside_plot=args.do_legend_outside_plot,
        )


if __name__ == "__main__":
    main()
