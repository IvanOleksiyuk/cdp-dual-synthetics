import os
import cv2
import numpy as np
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger(__name__)

import pyrootutils
import cv2 as cv
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

# local imports
from src.utils.default_datasets import image_datasets


def imgray(img_3c: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_3c, cv.COLOR_BGR2GRAY) if len(img_3c.shape) == 3 and img_3c.shape[2] == 3 else img_3c

def minmax(
    img: np.ndarray,
    vmin: int | None = None,
    vmax: int | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    vmin = vmin or img.min()
    vmax = vmax or img.max()
    return ((img - vmin) / (vmax - vmin)).astype(dtype)


def normalize_img(img: np.ndarray) -> np.ndarray:
    return (minmax(imgray(img)) * 255).astype(np.uint8).astype(np.float32)/255.0

def load_image(uid, dataset_name, shot=None, dataset_base_path="data/wifs2024dataset/wifs2024dataset", 
               target_size=(684, 684), interpolation="nearest", preserve_aspect_ratio=False,
               block=None, block_size=(128, 128), block_stride=(64, 64), structure=None):
    """
    Load an image from the dataset based on UID, dataset name, and optionally shot number.
    Then resize it to the target size using the specified interpolation method.
    Optionally extract blocks/patches from the image.
    
    Args:
        uid (str or int): Unique identifier of the image (e.g., "000145" or 145).
        dataset_name (str): Dataset identifier from image_datasets dict (e.g., "o76iP12" or "tem").
        shot (str or int, optional): View number (e.g., "0002" or 2). Required for non-template images.
        dataset_base_path (str): Base path to the dataset.
        target_size (tuple): Target size (width, height) for resizing, default (684, 684).
        interpolation (str): Interpolation method for resizing. Options:
            - "nearest": Nearest neighbor interpolation (fastest, lowest quality)
            - "linear": Bilinear interpolation
            - "cubic": Bicubic interpolation
            - "area": Area interpolation (good for downsampling)
            - "lanczos": Lanczos interpolation (best quality, slowest)
        preserve_aspect_ratio (bool): If True, preserves the aspect ratio and pads if necessary.
        block (None, int, or "all"): If None, return the whole image. If int, extract the specified block.
                                     If "all", return all blocks as a list.
        block_size (tuple): Size of blocks to extract (height, width), default (128, 128).
        block_stride (tuple): Stride for block extraction (vertical, horizontal), default (64, 64).
        structure (str, optional): Override the structure type. If set to "generated", will load 
                                  pre-generated blocks from dataset_base_path/dataset_name/uid/block_X.tiff
        
    Returns:
        np.ndarray or list of np.ndarray: The loaded and processed image(s) normalized to [0, 1].
    """
    # Format UID if needed (ensure it's a 6-digit string)
    if isinstance(uid, int):
        uid = f"{uid:06d}"
    
    # Handle generated blocks structure case
    if structure == "generated":
        if block is None:
            raise ValueError("Block parameter is required for 'generated' structure")
        
        # Get path to the UID directory
        uid_dir = Path(dataset_base_path) / dataset_name / uid
        
        if block == "all":
            # Find all block files for this UID
            block_files = sorted(uid_dir.glob("block_*.tiff"), key=lambda p: int(p.stem.split('_')[1]))
            
            if not block_files:
                raise FileNotFoundError(f"No block files found in {uid_dir}")
            
            # Load all blocks
            blocks = []
            for block_file in block_files:
                img = cv2.imread(str(block_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logger.warning(f"Failed to load block: {block_file}, skipping")
                    continue
                
                # Convert to float and normalize to [0, 1]
                blocks.append(img.astype("float32") / 255.0)
            
            logger.debug(f"Loaded {len(blocks)} blocks from {uid_dir}")
            return blocks
        else:
            # Construct path to the specific pre-generated block
            block_file = uid_dir / f"block_{block}.tiff"
            logger.debug(f"Loading pre-generated block from: {block_file}")
            
            # Check if the file exists
            if not block_file.exists():
                raise FileNotFoundError(f"Block file not found: {block_file}")
            
            # Load the block
            img = cv2.imread(str(block_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load block: {block_file}")
            
            # Convert to float and normalize to [0, 1]
            return img.astype("float32") / 255.0
    
    # Check if the dataset_name exists
    if dataset_name not in image_datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found in image_datasets")
    
    # Get dataset info
    dataset_info = image_datasets[dataset_name]
    rel_path = dataset_info["rel_path"]
    dataset_structure = structure if structure is not None else dataset_info["structure"]
    
    # Determine file path based on structure
    if dataset_structure == "template":
        # Template structure: {base_path}/{rel_path}/{uid}.tiff
        image_file = Path(dataset_base_path) / rel_path / f"{uid}.tiff"
        logger.debug(f"Loading template image from: {image_file}")  # Changed to debug
    elif dataset_structure == "default":
        # Default structure: {base_path}/{rel_path}/{uid}/{shot}.tiff
        if shot is None:
            raise ValueError("View parameter is required for non-template datasets")
        
        # Format shot if needed (ensure it's a 4-digit string)
        if isinstance(shot, int):
            shot = f"{shot:04d}"
            
        image_file = Path(dataset_base_path) / rel_path.lstrip('/') / uid / f"{shot}.tiff"
        logger.debug(f"Loading phone image from: {image_file}")  # Changed to debug
    else:
        raise ValueError(f"Unknown structure type '{dataset_structure}' for dataset '{dataset_name}'")
    
    # Check if the file exists
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file}")
    
    # Load the image
    img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load image: {image_file}")
    
    # Convert to float and normalize to [0, 1]
    img = img.astype("float32") / 255.0
    
    # Map interpolation method to OpenCV constant
    interpolation_methods = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4
    }
    
    if interpolation not in interpolation_methods:
        raise ValueError(f"Unknown interpolation method '{interpolation}'. Available options: {list(interpolation_methods.keys())}")
    
    interp = interpolation_methods[interpolation]
    
    # Get original dimensions
    original_height, original_width = img.shape[:2]
    
    # Resize image
    if preserve_aspect_ratio:
        # Calculate aspect ratio
        aspect = original_width / original_height
        target_width, target_height = target_size
        
        # Determine new dimensions while preserving aspect ratio
        if aspect > 1:  # Width > Height
            new_width = target_width
            new_height = int(target_width / aspect)
        else:  # Height >= Width
            new_height = target_height
            new_width = int(target_height * aspect)
            
        # Resize while preserving aspect ratio
        resized = cv2.resize(img, (new_width, new_height), interpolation=interp)
        
        # Create a blank canvas of the target size
        canvas = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Calculate position to paste the resized image (centered)
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        resized_img = canvas
    else:
        # Simple resize to target dimensions
        resized_img = cv2.resize(img, target_size, interpolation=interp)
    
    logger.debug(f"Resized image from {original_width}x{original_height} to {target_size[0]}x{target_size[1]} using {interpolation} interpolation")  # Changed to debug
    
    # Extract blocks if requested
    if block is None:
        # Return the entire image
        return resized_img
    
    # Get dimensions of the resized image
    img_height, img_width = resized_img.shape[:2]
    block_height, block_width = block_size
    stride_y, stride_x = block_stride
    
    # Calculate number of blocks in each dimension
    num_blocks_y = max(1, (img_height - block_height) // stride_y + 1)
    num_blocks_x = max(1, (img_width - block_width) // stride_x + 1)
    total_blocks = num_blocks_y * num_blocks_x
    
    logger.debug(f"Image will be divided into {total_blocks} blocks ({num_blocks_y}x{num_blocks_x}), " 
                f"each {block_width}x{block_height} pixels with stride {stride_x}x{stride_y}")  # Changed to debug
    
    # Function to extract a specific block by index
    def extract_block(block_idx):
        if block_idx >= total_blocks:
            raise ValueError(f"Block index {block_idx} is out of range. Image has {total_blocks} blocks.")
        
        # Convert block index to row, column coordinates
        block_y = block_idx // num_blocks_x
        block_x = block_idx % num_blocks_x
        
        # Calculate pixel coordinates
        y_start = block_y * stride_y
        x_start = block_x * stride_x
        y_end = min(y_start + block_height, img_height)
        x_end = min(x_start + block_width, img_width)
        
        # Extract the block
        return resized_img[y_start:y_end, x_start:x_end].copy()
    
    if block == "all":
        # Return all blocks as a list
        blocks = []
        for i in range(total_blocks): 
            blocks.append(normalize_img(extract_block(i)))
        logger.debug(f"Extracted all {total_blocks} blocks from the image")  # Changed to debug
        return blocks
    else:
        # Return a specific block
        try:
            block_idx = int(block)
            extracted_block = normalize_img(extract_block(block_idx))
            logger.debug(f"Extracted block {block_idx} from the image")  # Changed to debug
            return extracted_block
        except (ValueError, TypeError):
            raise ValueError(f"Invalid block parameter: {block}. Must be None, an integer, or 'all'")
        
        
        
if __name__ == "__main__":
    # Example usage
    uid = "000263"
    dataset_name = "o55iP12w"
    shot = "0001"
    block = 0
    dataset_base_path = "data/wifs2024dataset/wifs2024dataset"
    
    # Load the image
    img = load_image(uid, dataset_name, shot=shot, dataset_base_path=dataset_base_path, block=block)
    
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap="gray")
    plt.savefig("2.png")
    print("a")