from __future__ import annotations

#!/usr/bin/env python3
"""
Generate a LaTeX table from a jsonpickle-encoded summary produced by comparison scripts.

Input JSON structure example is provided in the user's prompt. This script:
- loads the JSON via jsonpickle
- extracts abs_values for metrics [MSE, PCC, SSIM]
- for AUC-like summaries, flips values < 0.5 to (1 - value) for readability
- builds a 2-group table (left/right) with rows:
  t, \hat{x}_{C1}, \hat{x}_{C2 \rightarrow C1}, x_{C2}, x_{C1}
- bolds the maximum per metric within each group

Usage:
  python jsonresilts2latex.py --input path/to/summary.json [--output table.tex] \
	  [--left-class iPXSo --right-class iP15m] \
	  [--left-header "iPhone XS wide" --right-header "iPhone 15 Pro macro"]

If --left-class/--right-class are not provided, the two dataset_pairs order in JSON is used
as left then right. Headers default to a small built-in mapping or class code strings.
"""

import argparse
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import jsonpickle


def load_summary(path: str) -> Dict[str, Any]:
	with open(path, 'r') as f:
		return jsonpickle.decode(f.read())
#!/usr/bin/env python3
"""
Generate a LaTeX table from a jsonpickle-encoded summary produced by comparison scripts.

Input example is shown in the prompt. This script:
- loads the JSON via jsonpickle
- extracts abs_values for metrics [MSE, PCC, SSIM]
- if metric_type == 'auc', flips values < 0.5 to (1 - value)
- builds a 2-group table with rows: t, \hat{x}_{C1}, \hat{x}_{C2 -> C1}, x_{C2}, x_{C1}
- bolds the max per metric within each group

Usage:
  python jsonresilts2latex.py --input path/to/summary.json [--output table.tex]
							  [--left-class iPXSo --right-class iP15m]
							  [--left-header "iPhone XS wide" --right-header "iPhone 15 Pro macro"]
"""

import argparse
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import jsonpickle


def load_summary(path: str) -> Dict[str, Any]:
	with open(path, "r") as f:
		return jsonpickle.decode(f.read())


def extract_device_code_from_pair(pair: str) -> str:
	m = re.search(r"o55([A-Za-z0-9]+)_vs", pair)
	if m:
		return m.group(1)
	m2 = re.search(r"_vs_f55([A-Za-z0-9]+)$", pair)
	return m2.group(1) if m2 else pair


DEFAULT_HEADER_MAP = {"iPXSo": "iPhone XS wide", "iP15m": "iPhone 15 Pro macro"}


def is_nan(x: Any) -> bool:
	try:
		return x is None or (isinstance(x, float) and math.isnan(x))
	except Exception:
		return True


def flip_auc_if_needed(val: Optional[float], flip: bool, cap: Optional[float] = 0.999) -> Optional[float]:
	if val is None or is_nan(val):
		return None
	v = float(val)
	if flip and v < 0.5:
		v = 1.0 - v
	if cap is not None and v > cap:
		v = cap
	return v


def build_metric_lookup(summary: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
	tables = summary.get("tables", [])
	lookup: Dict[str, Dict[str, Any]] = {}
	for t in tables:
		mname = t.get("metric")
		if isinstance(mname, str):
			lookup[mname] = t
	dataset_pairs = summary.get("dataset_pairs", [])
	return lookup, dataset_pairs


def get_abs_value(metric_lookup: Dict[str, Dict[str, Any]], metric: str, ref: str, pair_index: int) -> Optional[float]:
	table = metric_lookup.get(metric)
	if not table:
		return None
	refs = table.get("references", [])
	try:
		i = refs.index(ref)
	except ValueError:
		return None
	values = table.get("abs_values")
	if not values:
		return None
	try:
		v = values[i][pair_index]
	except Exception:
		return None
	try:
		vf = float(v)
		if math.isnan(vf):
			return None
		# Ensure non-negativity for PCC and SSIM by taking absolute value if negative
		if isinstance(metric, str) and metric.upper() in {"PCC", "SSIM"} and vf < 0:
			vf = abs(vf)
		return vf
	except Exception:
		return None


def format_number(v: Optional[float], decimals: int = 3) -> str:
	if v is None or is_nan(v):
		return ""
	return f"{v:.{decimals}f}"


def bold_max_per_column(rows: List[List[Optional[float]]]) -> List[List[bool]]:
	"""Deprecated helper kept for backward-compat; prefers maxima in all columns."""
	if not rows:
		return []
	n_cols = max(len(r) for r in rows)
	eps = 1e-12
	col_max: List[float] = []
	for c in range(n_cols):
		m = None
		for r in rows:
			if c >= len(r):
				continue
			v = r[c]
			if v is None or is_nan(v):
				continue
			m = v if m is None else max(m, v)
		col_max.append(m if m is not None else float("nan"))
	masks: List[List[bool]] = []
	for r in rows:
		mask: List[bool] = []
		for c in range(n_cols):
			v = r[c] if c < len(r) else None
			m = col_max[c]
			mask.append(False if (v is None or is_nan(v) or is_nan(m)) else (abs(v - m) <= eps))
		masks.append(mask)
	return masks


def bold_extreme_per_column(rows: List[List[Optional[float]]], prefer_max: List[bool]) -> List[List[bool]]:
	"""Return a boolean mask highlighting extremes per column.

	- prefer_max[c] = True  -> highlight max in column c
	- prefer_max[c] = False -> highlight min in column c
	"""
	if not rows:
		return []
	n_cols = max(len(r) for r in rows)
	eps = 1e-12
	col_extreme: List[float] = []
	for c in range(n_cols):
		values: List[float] = []
		for r in rows:
			if c >= len(r):
				continue
			v = r[c]
			if v is None or is_nan(v):
				continue
			values.append(v)
		if not values:
			col_extreme.append(float("nan"))
		else:
			col_extreme.append(max(values) if (prefer_max[c] if c < len(prefer_max) else True) else min(values))
	masks: List[List[bool]] = []
	for r in rows:
		mask: List[bool] = []
		for c in range(n_cols):
			v = r[c] if c < len(r) else None
			e = col_extreme[c]
			mask.append(False if (v is None or is_nan(v) or is_nan(e)) else (abs(v - e) <= eps))
		masks.append(mask)
	return masks


def main() -> None:
	ap = argparse.ArgumentParser(description="Convert jsonpickle summary to LaTeX table.")
	ap.add_argument("--input", required=True, help="Path to json file produced by summary tool")
	ap.add_argument("--output", default=None, help="Path to write LaTeX; defaults to stdout")
	ap.add_argument("--left-class", default=None, help="Device code for left group (e.g., iPXSo)")
	ap.add_argument("--right-class", default=None, help="Device code for right group (e.g., iP15m)")
	ap.add_argument("--left-header", default=None, help="Header label for left group")
	ap.add_argument("--right-header", default=None, help="Header label for right group")
	ap.add_argument("--decimals", type=int, default=3)
	ap.add_argument("--no_flip_auc", action="store_true", help="Do not flip values < 0.5 for AUC-like summaries")
	args = ap.parse_args()

	summary = load_summary(args.input)
	metric_type = str(summary.get("metric_type", "")).lower()
	flip_auc = (metric_type == "auc") and (not args.no_flip_auc)

	metric_lookup, dataset_pairs = build_metric_lookup(summary)
	if len(dataset_pairs) < 2:
		raise SystemExit("Expected at least 2 dataset_pairs in the summary")

	# Determine classes and indices
	codes = [extract_device_code_from_pair(p) for p in dataset_pairs]

	if args.left_class and args.right_class:
		try:
			left_idx = codes.index(args.left_class)
			right_idx = codes.index(args.right_class)
		except ValueError:
			raise SystemExit(f"Provided class codes not found in dataset_pairs: {codes}")
		left_code, right_code = args.left_class, args.right_class
	else:
		# Default ordering preference: iPXSo (iPhone XS) first, then iP15m (iPhone 15 Pro macro)
		if ("iPXSo" in codes) and ("iP15m" in codes):
			left_code, right_code = "iPXSo", "iP15m"
			left_idx, right_idx = codes.index(left_code), codes.index(right_code)
		else:
			left_code, right_code = codes[0], codes[1]
			left_idx, right_idx = 0, 1

	left_header = args.left_header or DEFAULT_HEADER_MAP.get(left_code, left_code)
	right_header = args.right_header or DEFAULT_HEADER_MAP.get(right_code, right_code)

	# Reference names expected in tables
	# tem and duplicates are fixed
	ref_tem = "tem"
	ref_dup = lambda code: f"o55{code}DUP"

	# Build a unified list of all reference names present in the tables
	all_references: List[str] = []
	seen = set()
	for t in metric_lookup.values():
		for r in t.get("references", []) or []:
			if isinstance(r, str) and r not in seen:
				seen.add(r)
				all_references.append(r)


	# Helper: find per-class roman baseline reference like
	#   reference_roman_..._class_o55{CODE}
	def find_roman_baseline_ref_for(code: str) -> Optional[str]:
		candidates = [
			r for r in all_references
			if r.startswith("reference_roman_") and (f"_class_o55{code}" in r)
		]
		return sorted(candidates, key=len)[0] if candidates else None

	# Helper: find per-class l2cond baseline reference like
	#   reference_l2cond_..._class_o55{CODE}
	def find_l2cond_baseline_ref_for(code: str) -> Optional[str]:
		candidates = [
			r for r in all_references
			if r.startswith("reference_l2cond_") and (f"_class_o55{code}" in r)
		]
		return sorted(candidates, key=len)[0] if candidates else None

	# Helper: find transport reference like
	#   reference_l2condtransport_maxinfo_..._from_o55{C2}_to_o55{C1}
	def find_transport_ref_from_to(from_code: str, to_code: str) -> Optional[str]:
		needle_from = f"_from_o55{from_code}_to_o55{to_code}"
		candidates = [
			r for r in all_references
			if r.startswith("reference_l2condtransport_maxinfo_") and (needle_from in r)
		]
		return sorted(candidates, key=len)[0] if candidates else None

	# Metrics order for columns per group
	metrics = ["MSE", "PCC", "SSIM"]


	# Build rows for each group separately, now with roman and l2cond baselines
	def group_values(c1_code: str, c2_code: str, pair_index: int) -> List[List[Optional[float]]]:
		rows: List[List[Optional[float]]] = []
		# 1) t
		row_t = [flip_auc_if_needed(get_abs_value(metric_lookup, m, ref_tem, pair_index), flip_auc, 0.999 if flip_auc else None) for m in metrics]
		rows.append(row_t)
		# 2) \hat{x}_{C1} roman
		ref_xc1hat_roman = find_roman_baseline_ref_for(c1_code)
		row_xc1hat_roman = [
			flip_auc_if_needed(
				get_abs_value(metric_lookup, m, ref_xc1hat_roman, pair_index),
				flip_auc,
				0.999 if flip_auc else None,
			)
			for m in metrics
		]
		rows.append(row_xc1hat_roman)
		# 3) \hat{x}_{C1} l2cond
		ref_xc1hat_l2cond = find_l2cond_baseline_ref_for(c1_code)
		row_xc1hat_l2cond = [
			flip_auc_if_needed(
				get_abs_value(metric_lookup, m, ref_xc1hat_l2cond, pair_index),
				flip_auc,
				0.999 if flip_auc else None,
			)
			for m in metrics
		]
		rows.append(row_xc1hat_l2cond)
		# 4) \hat{x}_{C2 -> C1}
		ref_transport = find_transport_ref_from_to(c2_code, c1_code)
		row_transport = [
			flip_auc_if_needed(
				get_abs_value(metric_lookup, m, ref_transport, pair_index),
				flip_auc,
				0.999 if flip_auc else None,
			)
			for m in metrics
		]
		rows.append(row_transport)
		# 5) x_{C2}
		row_xc2 = [flip_auc_if_needed(get_abs_value(metric_lookup, m, ref_dup(c2_code), pair_index), flip_auc, 0.999 if flip_auc else None) for m in metrics]
		rows.append(row_xc2)
		# 6) x_{C1}
		row_xc1 = [flip_auc_if_needed(get_abs_value(metric_lookup, m, ref_dup(c1_code), pair_index), flip_auc, 0.999 if flip_auc else None) for m in metrics]
		rows.append(row_xc1)
		return rows


	left_rows = group_values(left_code, right_code, left_idx)
	right_rows = group_values(right_code, left_code, right_idx)

	# Compute bold masks per group
	# Spec: Bold the best per metric among the first five rows (t, roman, l2cond, xhat_C2->C1, x_C2).
	def bold_mask_group(rows: List[List[Optional[float]]]) -> List[List[bool]]:
		if not rows:
			return []
		n_rows = len(rows)
		n_cols = max(len(r) for r in rows)
		mask = [[False for _ in range(n_cols)] for _ in range(n_rows)]
		# Bold maxima among first five rows (indices 0..4) per column
		main_rows = rows[:5]
		# Preference: for AUC we prefer max everywhere; for similarity metrics we prefer min for MSE, max for PCC/SSIM
		if metric_type == "auc":
			prefer_max = [True, True, True]
		else:
			prefer_max = [(m != "MSE") for m in metrics]
		main_mask = bold_extreme_per_column(main_rows, prefer_max)
		for i in range(min(5, n_rows)):
			for j in range(n_cols):
				if i < len(main_mask) and j < len(main_mask[i]) and main_mask[i][j]:
					mask[i][j] = True
		return mask

	left_bold = bold_mask_group(left_rows)
	right_bold = bold_mask_group(right_rows)

	# Row labels (roman, l2cond)
	row_labels = [
		r"$\mathbf{t}$",
		r"$\mathbf{\hat{x}}_{C1}\,\,[4]$",
		r"$\mathbf{\hat{x}}_{C1}\,\,L_2$",
		r"$\mathbf{\hat{x}}_{C2 \rightarrow C1}$",
		r"$\mathbf{x}_{C2},C_2\neq C_1$",
		r"$\mathbf{x}_{C1}$",
	]

	# Build LaTeX
	lines: List[str] = []
	lines.append(r"\begin{table*}[htbp]")
	lines.append(r"\centering")
	lines.append("")
	lines.append(r"% Adjust row height")
	lines.append(r"\renewcommand{\arraystretch}{1.3}")
	lines.append("")
	lines.append(r"\begin{tabular}{l|ccc|ccc}")
	lines.append(r"\toprule")
	lines.append(f" & \\multicolumn{{3}}{{c|}}{{{left_header}}} & \\multicolumn{{3}}{{c}}{{{right_header}}} \\\\")
	if metric_type == "auc":
		lines.append(r"Reference & $\text{AUC}_{\text{MSE}}$ & $\text{AUC}_{\text{PCC}}$ & $\text{AUC}_{\text{SSIM}}$ & $\text{AUC}_{\text{MSE}}$ & $\text{AUC}_{\text{PCC}}$ & $\text{AUC}_{\text{SSIM}}$ \\")
	else:
		lines.append(r"Reference & MSE$\downarrow$ & PCC$\uparrow$ & SSIM$\uparrow$ & MSE$\downarrow$ & PCC$\uparrow$ & SSIM$\uparrow$ \\")
	lines.append(r"\midrule")


	for i, label in enumerate(row_labels):
		left_vals = left_rows[i]
		right_vals = right_rows[i]
		left_fmt = [format_number(v, args.decimals) for v in left_vals]
		right_fmt = [format_number(v, args.decimals) for v in right_vals]

		# Apply bold
		left_fmt = [f"\\textbf{{{s}}}" if left_bold[i][j] and s else s for j, s in enumerate(left_fmt)]
		right_fmt = [f"\\textbf{{{s}}}" if right_bold[i][j] and s else s for j, s in enumerate(right_fmt)]

		row = (f"{label} & "
			+ " & ".join(left_fmt)
			+ " & "
			+ " & ".join(right_fmt)
			+ r" \\")
		lines.append(row)
		if i == 4:
			lines.append(r"\midrule")

	lines.append(r"\bottomrule")
	lines.append(r"\end{tabular}")
	if metric_type == "auc":
		lines.append(r"\caption{ROC AUC for original vs fake discrimination based on different similarity scores used between the reference and probe, using blocks of $64\times64$ pixels.}")
		lines.append(r"\label{tab:auc}")
	else:
		lines.append(r"\caption{Average similarity metrics between the original images and various references corresponding to them. The arrow shows the direction of higher similarity.}")
		lines.append(r"\label{tab:sim}")
	lines.append(r"%\vspace{-6mm}")
	lines.append(r"\end{table*}")

	out = "\n".join(lines) + "\n"
	if args.output:
		os.makedirs(os.path.dirname(args.output), exist_ok=True)
		with open(args.output, "w") as f:
			f.write(out)
		print(f"Wrote LaTeX table to {args.output}")
	else:
		print(out)


if __name__ == "__main__":
	main()
