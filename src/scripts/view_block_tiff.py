"""Quick viewer for a 64x64 TIFF block using matplotlib."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import pyrootutils
from skimage.metrics import structural_similarity as ssim

ROOT = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.data.data_new import CDPImage
from src.utils.default_datasets import image_datasets


EXPERIMENT_BASE = Path("workspace/2_device_run_1")

DEFAULT_SYNTHETIC_IP15M = Path(
    EXPERIMENT_BASE
    / "l2cond_EpsoniP15miP14wiPXSo/outputs/test_transformed/"
    / "reference_l2cond_EpsoniP15miP14wiPXSo_class_o55iP15m/"
    / "000203/block_10.tiff"
)

DEFAULT_SYNTHETIC_IPXSO = Path(
    EXPERIMENT_BASE
    / "l2cond_EpsoniP15miP14wiPXSo/outputs/test_transformed/"
    / "reference_l2cond_EpsoniP15miP14wiPXSo_class_o55iPXSo/"
    / "000203/block_10.tiff"
)

DEFAULT_DUAL_SYNTHETIC_IP15M_TO_IPXSO = Path(
    EXPERIMENT_BASE
    / "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo/"
    / "outputs/test_transformed/"
    / "reference_l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iP15m_to_o55iPXSo/"
    / "000203/block_10.tiff"
)

DEFAULT_DUAL_SYNTHETIC_IPXSO_TO_IP15M = Path(
    EXPERIMENT_BASE
    / "l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo/"
    / "outputs/test_transformed/"
    / "reference_l2condtransport_maxinfo_MR50_EpsoniP15miP14wiPXSo_from_o55iPXSo_to_o55iP15m/"
    / "000203/block_10.tiff"
)

DEFAULT_DATASET_BASE = Path("data/wifs2024dataset/wifs2024dataset")
DEFAULT_UID = "000203"
DEFAULT_SHOT = "0001"
DEFAULT_BLOCK_ID = 10
DEFAULT_TEMPLATE_DATASET = "tem"
DEFAULT_IPXSO_DATASET = "o55iPXSo"
DEFAULT_IP15M_DATASET = "o55iP15m"


def _load_block_from_dataset(
    dataset_name: str,
    uid: str,
    block_id: int,
    dataset_base_path: Path,
    shot: str | None = None,
) -> np.ndarray:
    if dataset_name not in image_datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = image_datasets[dataset_name]
    rel_path = dataset_info["rel_path"]
    structure = dataset_info["structure"]

    if structure == "template":
        image_file_path = dataset_base_path / rel_path / f"{uid}.tiff"
        shot_name = "template"
    else:
        if shot is None:
            raise ValueError(f"Shot is required for dataset {dataset_name}")
        shot_name = f"{int(shot):04d}" if str(shot).isdigit() else str(shot)
        image_file_path = dataset_base_path / rel_path.lstrip("/") / uid / f"{shot_name}.tiff"

    if not image_file_path.exists():
        raise FileNotFoundError(f"Image not found: {image_file_path}")

    cdp_image = CDPImage(
        height=684,
        width=684,
        crinfo=f"{dataset_name}_{structure}",
        uid=uid,
        shot=shot_name,
        image_file_path=str(image_file_path),
        image_settings={"h": 684, "w": 684, "interpolation": "nearest"},
        block_settings={"block_h": 64, "block_w": 64, "stride_h": 64, "stride_w": 64},
        mode="image_file",
    )

    return cdp_image.get_block(block_id)


def _show_image(ax, img, title: str) -> None:
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def _to_float(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return _to_float(arr)


def _diff_image(probe: np.ndarray, reference: np.ndarray) -> np.ndarray:
    probe_f = _to_gray_float(probe)
    ref_f = _to_gray_float(reference)
    return ref_f - probe_f


def _metrics(probe: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    probe_f = _to_gray_float(probe)
    ref_f = _to_gray_float(reference)
    mse = float(np.mean((ref_f - probe_f) ** 2))
    pcc = (
        1.0 * float(np.corrcoef(ref_f.flatten(), probe_f.flatten())[0, 1])
        if probe_f.size and ref_f.size
        else float("nan")
    )
    ssim_val = 1.0 * float(ssim(ref_f, probe_f, data_range=1.0))
    return {"MSE": mse, "PCC": pcc, "SSIM": ssim_val}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a TIFF image with matplotlib.")
    parser.add_argument(
        "synthetic_iP15m",
        nargs="?",
        type=Path,
        default=DEFAULT_SYNTHETIC_IP15M,
        help="Path to the synthetic_iP15m TIFF (default: block_10.tiff)",
    )
    parser.add_argument(
        "--synthetic-iPXSo",
        dest="synthetic_iPXSo",
        type=Path,
        default=DEFAULT_SYNTHETIC_IPXSO,
        help="Path to the synthetic_iPXSo TIFF",
    )
    parser.add_argument(
        "--dual-synthetic",
        dest="dual_synthetic",
        type=Path,
        default=DEFAULT_DUAL_SYNTHETIC_IP15M_TO_IPXSO,
        help="Path to the dual-synthetic iP15m->iPXSo TIFF",
    )
    parser.add_argument(
        "--dual-synthetic-rev",
        dest="dual_synthetic_rev",
        type=Path,
        default=DEFAULT_DUAL_SYNTHETIC_IPXSO_TO_IP15M,
        help="Path to the dual-synthetic iPXSo->iP15m TIFF",
    )
    parser.add_argument(
        "--dataset-base",
        type=Path,
        default=DEFAULT_DATASET_BASE,
        help="Base path to the real/template datasets",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=DEFAULT_UID,
        help="UID to load for template and real images",
    )
    parser.add_argument(
        "--shot",
        type=str,
        default=DEFAULT_SHOT,
        help="Shot to load for real images (default: 0001)",
    )
    parser.add_argument(
        "--block-id",
        type=int,
        default=DEFAULT_BLOCK_ID,
        help="Block id to extract from real/template images",
    )
    parser.add_argument(
        "--template-dataset",
        type=str,
        default=DEFAULT_TEMPLATE_DATASET,
        help="Dataset name for template image",
    )
    parser.add_argument(
        "--ipxso-dataset",
        type=str,
        default=DEFAULT_IPXSO_DATASET,
        help="Dataset name for real iPXSo image",
    )
    parser.add_argument(
        "--ip15m-dataset",
        type=str,
        default=DEFAULT_IP15M_DATASET,
        help="Dataset name for real iP15m image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_iP15m_path: Path = args.synthetic_iP15m
    synthetic_iPXSo_path: Path = args.synthetic_iPXSo
    dual_synthetic_path: Path = args.dual_synthetic
    dual_synthetic_rev_path: Path = args.dual_synthetic_rev
    dataset_base: Path = args.dataset_base
    uid: str = args.uid
    shot: str = args.shot
    block_id: int = args.block_id
    template_dataset: str = args.template_dataset
    ipxso_dataset: str = args.ipxso_dataset
    ip15m_dataset: str = args.ip15m_dataset

    for candidate in (
        synthetic_iP15m_path,
        synthetic_iPXSo_path,
        dual_synthetic_path,
        dual_synthetic_rev_path,
    ):
        if not candidate.exists():
            raise FileNotFoundError(f"Image not found: {candidate}")

    with (
        Image.open(synthetic_iP15m_path) as img_a,
        Image.open(synthetic_iPXSo_path) as img_b,
        Image.open(dual_synthetic_path) as img_c,
        Image.open(dual_synthetic_rev_path) as img_d,
    ):
        data_a = np.asarray(img_a.convert("L"))
        data_b = np.asarray(img_b.convert("L"))
        data_c = np.asarray(img_c.convert("L"))
        data_d = np.asarray(img_d.convert("L"))

    template_block = _load_block_from_dataset(
        template_dataset, uid=uid, block_id=block_id, dataset_base_path=dataset_base
    )
    ipxso_block = _load_block_from_dataset(
        ipxso_dataset, uid=uid, shot=shot, block_id=block_id, dataset_base_path=dataset_base
    )
    ip15m_block = _load_block_from_dataset(
        ip15m_dataset, uid=uid, shot=shot, block_id=block_id, dataset_base_path=dataset_base
    )

    # Build rows: first 3 rows probe = iPXSo block, next 3 rows probe = iP15m block.
    probes = [ipxso_block] * 3 + [ip15m_block] * 3
    references = [
        template_block,          # template for iPXSo probe
        data_b,                  # synthetic iPXSo
        data_c,                  # dual iP15m->iPXSo (should be reference for iPXSo probe)
        template_block,          # template for iP15m probe
        data_a,                  # synthetic iP15m
        data_d,                  # dual iPXSo->iP15m (should be reference for iP15m probe)
    ]

    titles_probe = ["probe iPXSo", "", "", "probe iP15m", "", ""]
    titles_ref = [
        r"$\mathbf{t}$",
        r"$\hat{\mathbf{x}}_{\mathrm{iPXSw}}\,L_2$",
        r"$\hat{\mathbf{x}}_{\mathrm{iP15m \to iPXSo}}$",
        r"$\mathbf{t}$",
        r"$\hat{\mathbf{x}}_{\mathrm{iP15m}}\,L_2$",
        r"$\hat{\mathbf{x}}_{\mathrm{iPXSo \to iP15m}}$",
    ]

    diffs = [_diff_image(p, r) for p, r in zip(probes, references)]
    max_abs = max(max(abs(d.min()), abs(d.max())) for d in diffs) or 1.0

    fig, axes = plt.subplots(6, 4, figsize=(5, 9))
    diff_title_set = False
    for row in range(6):
        probe = probes[row]
        ref = references[row]
        diff = diffs[row]

        _show_image(axes[row, 0], probe, titles_probe[row])
        _show_image(axes[row, 1], ref, titles_ref[row])
        axes[row, 2].imshow(diff, cmap="bwr", vmin=-max_abs, vmax=max_abs)
        if not diff_title_set:
            axes[row, 2].set_title("diff")
            diff_title_set = True
        axes[row, 2].axis("off")

        metrics = _metrics(probe, ref)
        txt = f"MSE: {metrics['MSE']:.3g}\nPCC: {metrics['PCC']:.3g}\nSSIM: {metrics['SSIM']:.3g}"
        axes[row, 3].axis("off")
        axes[row, 3].text(0.05, 0.5, txt, va="center", ha="left", fontsize=10, family="monospace")

    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    output_path = EXPERIMENT_BASE / "plots" / "example.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
