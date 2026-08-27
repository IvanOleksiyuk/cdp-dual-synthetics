#!/usr/bin/env python3
"""
merge_latex_tables.py  (fixed)

Merges two LaTeX tables as described previously. This version fixes a parsing
bug that could drop the first data row after a \midrule and the last row before
\bottomrule (e.g., rows like $\mathbf{t}$ and $\mathbf{x}_{C1}$).

Key fixes:
- Strip \toprule, \midrule, \bottomrule anywhere in the tabular body BEFORE
  splitting rows, so control markers don't cause us to skip valid data rows.
- Keep behavior of removing redundant "AUC_" prefixes in AUC headers.

Usage:
  python merge_latex_tables.py --table1 table1.tex --table2 table2.tex --out merged_table.tex
"""

import argparse
import re
from pathlib import Path

TABULAR_BEGIN_RE = re.compile(r'\\begin\{tabular\}\{.*?\}', re.DOTALL)
TABULAR_END_RE = re.compile(r'\\end\{tabular\}')

def extract_tabular(tex: str) -> str:
    """Return the first \begin{tabular}...\end{tabular} block from the input."""
    m_begin = TABULAR_BEGIN_RE.search(tex)
    if not m_begin:
        raise ValueError("Could not find \\begin{tabular}{...} in input.")
    m_end = TABULAR_END_RE.search(tex, m_begin.end())
    if not m_end:
        raise ValueError("Found \\begin{tabular} but not matching \\end{tabular}.")
    return tex[m_begin.start():m_end.end()]

def clean_tabular_body(tabular_block: str) -> str:
    """Remove begin/end wrapper and strip control rules anywhere in the body."""
    # Cut to inside of tabular
    inner = re.sub(r'^.*?\\begin\{tabular\}\{[^\}]*\}', '', tabular_block, flags=re.DOTALL)
    inner = re.sub(r'\\end\{tabular\}.*$', '', inner, flags=re.DOTALL)

    # Remove top/mid/bottom rules no matter where they appear
    inner = re.sub(r'\\(toprule|midrule|bottomrule)\s*', '', inner)

    # Also normalize lines by removing trailing comments (keep values before '%')
    inner = re.sub(r'(?m)%.*$', '', inner)

    return inner.strip()

def split_rows(inner_body: str):
    """Split a cleaned tabular body into rows at unescaped '\\' row terminators."""
    parts = re.split(r'(?<!\\)\\\\', inner_body)
    rows = [p.strip() for p in parts if p.strip()]
    return rows

def parse_data_rows(tabular_block: str):
    """
    Parse the tabular block and return:
      headers_row: the row starting with 'Reference & ...'
      data: list of (reference_label, cells:list[str])
      colspec: the {l|ccc|ccc} string (if present)
    """
    # Extract column spec (for potential validation or reuse)
    m = re.search(r'\\begin\{tabular\}\{([^\}]*)\}', tabular_block)
    colspec = m.group(1).strip() if m else None

    inner = clean_tabular_body(tabular_block)
    raw_rows = split_rows(inner)

    headers_row = None
    data = []
    for row in raw_rows:
        # Skip multicolumn title rows except the "Reference & ..." header
        if r'\multicolumn' in row and 'Reference' not in row:
            continue

        row_clean = row.strip()
        # Header
        if row_clean.lstrip().startswith('Reference &'):
            headers_row = row_clean
            continue

        # Data (ampersand-delimited cells)
        if '&' in row_clean:
            # Split on unescaped & (simple case; LaTeX usually doesn't escape & in numbers/math here)
            cells = [c.strip() for c in row_clean.split('&')]
            if len(cells) < 2:
                continue
            ref = cells[0]
            data_cells = cells[1:]
            data.append((ref, data_cells))

    if headers_row is None:
        raise ValueError("Could not find a 'Reference &' header row in the tabular.")
    return headers_row, data, colspec

def build_index(data_rows):
    order = [ref for ref, _ in data_rows]
    idx = {ref: i for i, (ref, _) in enumerate(data_rows)}
    return order, idx

def merge_tables(sim_rows, auc_rows, device_titles, arraystretch=1.3, label="tab:sim_auc",
                 caption=("Average similarity metrics and ROC AUCs (original vs. fake discrimination) "
                          "computed on blocks of $64\\times64$ pixels between the original images and "
                          "various corresponding references. Within each device, similarity metrics "
                          "(MSE$\\downarrow$, PCC$\\uparrow$, SSIM$\\uparrow$) are listed first, "
                          "followed by the ROC AUCs based on those metrics. The arrow reflects superior "
                          "performance for the similarity metrics. For each reference, $C_1$ denotes the "
                          "device at the top of the column and $C_2$ denotes the device from the opposite "
                          "side of the table.")):
    """
    sim_rows: list[(ref, [xs_MSE,xs_PCC,xs_SSIM, p15_MSE,p15_PCC,p15_SSIM])]
    auc_rows: list[(ref, [xs_AUC_MSE,xs_AUC_PCC,xs_AUC_SSIM, p15_AUC_MSE,p15_AUC_PCC,p15_AUC_SSIM])]
    device_titles: (left_title, right_title)
    """
    # Validate references align
    sim_order, sim_idx = build_index(sim_rows)
    auc_order, auc_idx = build_index(auc_rows)
    if sim_order != auc_order:
        # Offer more helpful error
        missing_from_auc = [r for r in sim_order if r not in auc_idx]
        missing_from_sim = [r for r in auc_order if r not in sim_idx]
        raise ValueError(
            "Reference row order mismatch between similarity and AUC tables.\n"
            f"Only in similarity: {missing_from_auc}\n"
            f"Only in AUC: {missing_from_sim}"
        )

    left_title, right_title = device_titles

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append("")
    lines.append(r"% Adjust row height")
    lines.append(rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}")
    lines.append("")
    lines.append(r"\begin{tabular}{l|ccc|ccc|ccc|ccc}")
    lines.append(r"\toprule")
    lines.append(f" & \\multicolumn{{6}}{{c|}}{{{left_title}}} & \\multicolumn{{6}}{{c}}{{{right_title}}} \\\\")
    lines.append(r" & \multicolumn{3}{c|}{Similarity} & \multicolumn{3}{c|}{AUC$\uparrow$} & \multicolumn{3}{c|}{Similarity} & \multicolumn{3}{c}{ROC AUC} \\")
    lines.append(r"Reference & MSE$\downarrow$ & PCC$\uparrow$ & SSIM$\uparrow$ & MSE & PCC & SSIM & MSE$\downarrow$ & PCC$\uparrow$ & SSIM$\uparrow$ & MSE & PCC & SSIM \\")
    lines.append(r"\midrule")

    row_lines = []
    for ref in sim_order:
        s_cells = sim_rows[sim_idx[ref]][1]
        a_cells = auc_rows[auc_idx[ref]][1]
        if len(s_cells) != 6 or len(a_cells) != 6:
            raise ValueError(f"Expected 6 similarity and 6 AUC cells per row for '{ref}'. Got {len(s_cells)} and {len(a_cells)}.")
        xs_sim = s_cells[:3]
        p15_sim = s_cells[3:]
        xs_auc = a_cells[:3]
        p15_auc = a_cells[3:]
        row = " & ".join([ref] + xs_sim + xs_auc + p15_sim + p15_auc) + r" \\"
        row_lines.append(row)

    # If the last reference is the "oracle" row ($\mathbf{x}_{C1}$), insert a midrule before it
    if row_lines and (sim_order[-1].replace(' ', '') == r'$\mathbf{x}_{C1}$'.replace(' ', '')):
        row_lines = row_lines[:-1] + [r"\midrule"] + [row_lines[-1]]

    lines.extend(row_lines)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)

def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')

def parse_table(path: Path):
    tex = load_text(path)
    tab = extract_tabular(tex)
    header, data, _ = parse_data_rows(tab)
    return header, data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table1", required=True, help="Path to LaTeX file containing Table 1 (Similarity metrics).")
    ap.add_argument("--table2", required=True, help="Path to LaTeX file containing Table 2 (AUCs).")
    ap.add_argument("--out", required=True, help="Output path for merged LaTeX table.")
    ap.add_argument("--label", default="tab:sim_auc", help="LaTeX label for the merged table.")
    ap.add_argument("--caption", default=None, help="LaTeX caption for the merged table.")
    ap.add_argument("--xs_title", default="iPhone XS wide", help="Left device title.")
    ap.add_argument("--p15_title", default="iPhone 15 Pro macro", help="Right device title.")
    ap.add_argument("--arraystretch", default="1.3", help="\\arraystretch value.")
    args = ap.parse_args()

    # Parse both tables
    _, sim_rows = parse_table(Path(args.table1))
    _, auc_rows = parse_table(Path(args.table2))

    caption = (args.caption if args.caption is not None else
               "Average similarity metrics and ROC AUCs (original vs. fake discrimination) "
               "computed on blocks of $64\\times64$ pixels between the original images and "
               "various corresponding references. Within each device, similarity metrics "
               "(MSE$\\downarrow$, PCC$\\uparrow$, SSIM$\\uparrow$) are listed first, followed by "
               "the ROC AUCs based on those metrics. The arrow reflects superior performance for "
               "the similarity metrics. For each reference, $C_1$ denotes the device at the top "
               "of the column and $C_2$ denotes the device from the opposite side of the table.")

    merged = merge_tables(
        sim_rows=sim_rows,
        auc_rows=auc_rows,
        device_titles=(args.xs_title, args.p15_title),
        arraystretch=args.arraystretch,
        label=args.label,
        caption=caption,
    )

    out_path = Path(args.out)
    out_path.write_text(merged, encoding='utf-8')
    print(f"Merged table written to: {out_path}")

if __name__ == "__main__":
    main()
