import os
from pathlib import Path
from typing import Tuple
import cv2
from tqdm import tqdm


def resize_image(image_path: Path, target_size: Tuple[int, int]) -> None:
    """
    Resize an image to the target size and overwrite the original image.

    Args:
        image_path (Path): Path to the image file.
        target_size (Tuple[int, int]): Target size (width, height) for resizing.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(image_path), resized_img)


def split_image_into_blocks(image_path: Path, block_size: Tuple[int, int], step: int) -> dict:
    """
    Split an image into blocks of a given size with a specified step.

    Args:
        image_path (Path): Path to the image file.
        block_size (Tuple[int, int]): Size of each block (height, width).
        step (int): Step size for sliding window.

    Returns:
        dict: A dictionary where keys are block indices (row, col) and values are the block images.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    blocks = {}

    block_idx = 0
    for y in range(0, h - block_size[0] + 1, step):
        for x in range(0, w - block_size[1] + 1, step):
            block = img[y:y + block_size[0], x:x + block_size[1]]
            blocks[block_idx] = block
            block_idx += 1

    return blocks


def process_dataset(
    dataset_path: str,
    target_size: Tuple[int, int] = (684, 684),
    block_size: Tuple[int, int] = (128, 128),
    step: int = 64,
    delete_full_image: bool = False,
) -> None:
    """
    Process the dataset by resizing each image, splitting it into blocks, and organizing them into subfolders.

    Args:
        dataset_path (str): Path to the dataset.
        target_size (Tuple[int, int]): Target size (width, height) for resizing.
        block_size (Tuple[int, int]): Size of each block (height, width).
        step (int): Step size for sliding window.
        delete_full_image (bool): Whether to delete the original full images after processing.
    """
    dataset_path = Path(dataset_path)

    # Iterate over train, valid, and test folders
    for split in ["train", "valid", "test"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
        # Iterate over each image ID folder with progress bar
        id_folders = [f for f in split_path.iterdir() if f.is_dir()]
        for id_folder in tqdm(id_folders, desc=f"{split} IDs", unit="id"):
            ## Create a blocks directory for this ID
            blocks_dir = id_folder / "blocks"
            blocks_dir.mkdir(parents=True, exist_ok=True)

            # Collect all images for this ID
            images = list(id_folder.glob("*.tiff"))
            if not images:
                continue

            # Resize and split each image into blocks
            block_folders = {}
            for image_path in images:
                # Resize the image to the target size
                resize_image(image_path, target_size)

                # Split the resized image into blocks
                image_name = image_path.stem
                blocks = split_image_into_blocks(image_path, block_size, step)

                # Save each block into a corresponding subfolder
                for block_idx, block in blocks.items():
                    block_folder = blocks_dir / f"block_{block_idx}"
                    block_folder.mkdir(parents=True, exist_ok=True)

                    block_file = block_folder / f"{image_name}.tiff"
                    cv2.imwrite(str(block_file), block)

                    # Track block folders for cleanup
                    block_folders[block_folder] = True

            # Optionally delete the full images
            if delete_full_image:
                for image_path in images:
                    image_path.unlink()

    print(f"Dataset processed and resized to {target_size}, then split into blocks with size {block_size} and step {step}.")


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a dataset by resizing images and splitting them into blocks.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--target_size", type=str, default="684,684", help="Target size for resizing (width,height).")
    parser.add_argument("--block_size", type=str, default="128,128", help="Block size (height,width).")
    parser.add_argument("--step", type=int, default=64, help="Step size for sliding window.")
    parser.add_argument("--delete_full_image", action="store_true", help="Delete full images after processing.")

    args = parser.parse_args()

    # Parse tuple arguments
    target_size = tuple(map(int, args.target_size.split(',')))
    block_size = tuple(map(int, args.block_size.split(',')))

    # Process the dataset
    process_dataset(
        dataset_path=args.dataset_path,
        target_size=target_size,
        block_size=block_size,
        step=args.step,
        delete_full_image=args.delete_full_image,
    )

#python src/helper_scripts/blockify_dataset.py --dataset_path=data/cdp_transit_debug/
#python src/helper_scripts/blockify_dataset.py --dataset_path=data/cdp_transport_dataset_default/
#python src/helper_scripts/blockify_dataset.py --dataset_path=data/cdp_transport_dataset_default/