import os
import shutil
from pathlib import Path
from typing import Dict

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.utils.default_datasets import image_datasets


def restructure_dataset(
    original_dataset_path: str,
    path_to_new_dataset: str,
    device_classes: Dict[str, str],
    train_range: tuple = None,
    valid_range: tuple = None,
    test_range: tuple = None,
    save_if_any: bool = False,  # New parameter to control saving behavior
) -> None:
    """
    Reshape the dataset by creating a new dataset structure with templates and corresponding images.

    Args:
        original_dataset_path (str): Path to the original dataset.
        path_to_new_dataset (str): Path to the new dataset to be created.
        device_classes (Dict[str, str]): Dictionary where keys are specific device class paths relative to the original dataset,
                                         and values are the custom tag names for each class.
        train_range (tuple, optional): Range of IDs for the training set (start, end).
        valid_range (tuple, optional): Range of IDs for the validation set (start, end).
        test_range (tuple, optional): Range of IDs for the test set (start, end).
        save_if_any (bool, optional): If True, save IDs if at least one device has images. Default is False (all devices must have images).
    """
    # Create the new dataset directory
    new_dataset_path = Path(path_to_new_dataset)
    new_dataset_path.mkdir(parents=True, exist_ok=True)

    # Paths for the template and device classes
    template_path = Path(original_dataset_path) / "orig_template" / "rcod"
    device_paths = {device: Path(original_dataset_path) / device for device in device_classes.keys()}

    # Initialize counters and ID storage
    total_ids = 0
    ids_with_images = []

    # Iterate over template images
    for template_file in template_path.glob("*.tiff"):
        template_id = template_file.stem  # Extract the ID from the filename (e.g., "000001")

        # Check if corresponding images exist for all or any device classes
        device_images = {}
        for device, device_path in device_paths.items():
            device_rcod_path = device_path / template_id
            if device_rcod_path.exists() and any(device_rcod_path.glob("*.tiff")):
                # Take the first image for simplicity
                device_images[device] = next(device_rcod_path.glob("*.tiff"))

        # Save the ID based on the save_if_any flag
        if (save_if_any and len(device_images) > 0) or (not save_if_any and len(device_images) == len(device_classes)):
            total_ids += 1
            ids_with_images.append(template_id)

            # Create a subdirectory for this ID in the new dataset
            id_dir = new_dataset_path / template_id
            id_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the template image
            shutil.copyfile(template_file, id_dir / f"{template_id}_t.tiff")

            # Copy one image per device class
            for device, image_path in device_images.items():
                tag = device_classes[device]  # Use the custom tag for the device class
                shutil.copyfile(image_path, id_dir / f"{template_id}_{tag}.tiff")

    print("ids_with_images:", ids_with_images)
    # Filter IDs into train, valid, and test sets based on provided ranges
    def filter_ids_by_range(ids, id_range):
        start, end = id_range
        return [id_ for id_ in ids if start <= int(id_) <= end]

    print("DEBUG: Splitting into train, valid, and test sets...")
    if train_range and valid_range and test_range:
        train_ids = filter_ids_by_range(ids_with_images, train_range)
        valid_ids = filter_ids_by_range(ids_with_images, valid_range)
        test_ids = filter_ids_by_range(ids_with_images, test_range)
    else:
        train_split = int(0.6 * total_ids)
        valid_split = int(0.8 * total_ids)
        train_ids = ids_with_images[:train_split]
        valid_ids = ids_with_images[train_split:valid_split]
        test_ids = ids_with_images[valid_split:]

    print("DEBUG: Copying to train, valid, and test directories...")
    # Create train, valid, and test directories
    for split_name, split_ids in zip(["train", "valid", "test"], [train_ids, valid_ids, test_ids]):
        split_dir = new_dataset_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for id_ in split_ids:
            id_dir = new_dataset_path / id_
            if id_dir.exists():  # Ensure the directory exists before copying
                # Copy the directory to the split directory
                shutil.copytree(id_dir, split_dir / id_)
            else:
                print(f"Warning: Directory {id_dir} does not exist and will be skipped.")

    print("DEBUG: Deleting original directories...")     
    # Delete the original directories after copying
    for id_ in ids_with_images:
        id_dir = new_dataset_path / id_
        if id_dir.exists():
            shutil.rmtree(id_dir)

    print(f"Total IDs in the new dataset: {total_ids}")
    print(f"Train IDs: {len(train_ids)}, Valid IDs: {len(valid_ids)}, Test IDs: {len(test_ids)}")


if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Restructure dataset into train, valid, and test sets.")
    parser.add_argument("--original_dataset_path", type=str, help="Path to the original dataset.")
    parser.add_argument("--path_to_new_dataset", type=str, help="Path to the new dataset to be created.")
    parser.add_argument("--train_range", type=str, help="Range of IDs for the training set (start,end).")
    parser.add_argument("--valid_range", type=str, help="Range of IDs for the validation set (start,end).")
    parser.add_argument("--test_range", type=str, help="Range of IDs for the test set (start,end).")
    parser.add_argument("--default", action="store_true", help="Run with default parameters.")
    parser.add_argument("--save_if_any", action="store_true", help="Save IDs if at least one device has images.")

    args = parser.parse_args()

    # Default parameters
    if args.default:
        original_dataset_path = "data/wifs2024dataset/wifs2024dataset" #data/
        path_to_new_dataset = "data/cdp_transport_dataset_default"
        train_range = (145, 192)
        valid_range = (193, 202)
        test_range = (203, 288)
    else:
        # Use provided arguments
        original_dataset_path = args.original_dataset_path
        path_to_new_dataset = args.path_to_new_dataset
        train_range = tuple(map(int, args.train_range.split(','))) if args.train_range else None
        valid_range = tuple(map(int, args.valid_range.split(','))) if args.valid_range else None
        test_range = tuple(map(int, args.test_range.split(','))) if args.test_range else None

    # Validate required arguments when not using default
    if not args.default and (not original_dataset_path or not path_to_new_dataset):
        raise ValueError("You must provide --original_dataset_path and --path_to_new_dataset unless --default is specified.")

    # Create device_classes dictionary from image_datasets
    device_classes = {
        dataset_info["rel_path"][1:]: dataset_name
        for dataset_name, dataset_info in image_datasets.items()
        if dataset_info["structure"] == "default"
    }

    # Restructure the dataset
    restructure_dataset(
        original_dataset_path=original_dataset_path,
        path_to_new_dataset=path_to_new_dataset,
        device_classes=device_classes,
        train_range=train_range,
        valid_range=valid_range,
        test_range=test_range,
        save_if_any=True
    )
    
# Run the scipt twice to create full and debug versions:
# DATA_DIR="data/"
# python src/helper_scripts/restructure_dataset.py --default
# python src/helper_scripts/restructure_dataset.py --original_dataset_path ${DATA_DIR}wifs2024dataset/wifs2024dataset \
#  --path_to_new_dataset ${DATA_DIR}cdp_transit_debug\
#  --train_range 145,147\
#  --valid_range 148,150\
#  --test_range 148,150

# Wyner
# DATA_DIR="data/"
# python src/helper_scripts/restructure_dataset.py --original_dataset_path ${DATA_DIR}wifs2024dataset/wifs2024dataset \
#  --path_to_new_dataset ${DATA_DIR}wifs2024dataset/cdp_transport_dataset_default\
#  --train_range 145,192\
#  --valid_range 193,202\
#  --test_range 203,288