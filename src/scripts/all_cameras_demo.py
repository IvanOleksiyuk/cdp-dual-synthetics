from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyrootutils

ROOT = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.data.data_new import CDPImage
from src.utils.default_datasets import image_datasets

DEFAULT_DATASET_BASE = Path("data/wifs2024dataset/wifs2024dataset")
DEFAULT_UID = "000203"
DEFAULT_SHOT = "0001"
DEFAULT_BLOCK_ID = 10
DEFAULT_TEMPLATE_DATASET = "tem"
O55_CAMERAS = [
    "o55Epson",
    "o55iPXSo",
    "o55iP12w",
    "o55iP15w",
    "o55iP14w",
    "o55iP15m",
    "o55iP14m",
]

HUMAN_TITLES = {
    "tem": "Digital\ntemplate",
    "o55Epson": "Epson\nscanner",
    "o55iP12w": "iPhone 12\nwide",
    "o55iPXSo": "iPhone XS\nwide",
    "o55iP15m": "iPhone 15\nmacro",
    "o55iP15w": "iPhone 15\nwide",
    "o55iP14w": "iPhone 14\nwide",
    "o55iP14m": "iPhone 14\nmacro",
}


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


def _show_image(ax, img: np.ndarray, title: str) -> None:
    arr = np.asarray(img)
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray")
    else:
        ax.imshow(arr)
    ax.set_title(title, fontsize=18)
    ax.axis("off")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show template + all o55 camera blocks in one row.")
    parser.add_argument("--dataset-base", type=Path, default=DEFAULT_DATASET_BASE, help="Base path to the dataset root")
    parser.add_argument("--uid", type=str, default=DEFAULT_UID, help="UID to load")
    parser.add_argument("--shot", type=str, default=DEFAULT_SHOT, help="Shot number for camera images (default: 0001)")
    parser.add_argument("--block-id", type=int, default=DEFAULT_BLOCK_ID, help="Block id to extract (0-indexed)")
    parser.add_argument(
        "--template-dataset",
        type=str,
        default=DEFAULT_TEMPLATE_DATASET,
        help="Dataset name for the digital template",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_base: Path = args.dataset_base
    uid: str = args.uid
    shot: str = args.shot
    block_id: int = args.block_id
    template_dataset: str = args.template_dataset

    datasets = [template_dataset] + O55_CAMERAS
    blocks: list[np.ndarray] = []
    titles: list[str] = []

    for name in datasets:
        block = _load_block_from_dataset(
            name,
            uid=uid,
            block_id=block_id,
            dataset_base_path=dataset_base,
            shot=None if name == template_dataset else shot,
        )
        blocks.append(block)
        titles.append(HUMAN_TITLES.get(name, name))

    fig, axes = plt.subplots(1, len(blocks), figsize=(2.2 * len(blocks), 2.4))
    for ax, block, title in zip(axes, blocks, titles):
        _show_image(ax, block, title)

    plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    plt.show()


if __name__ == "__main__":
    main()
