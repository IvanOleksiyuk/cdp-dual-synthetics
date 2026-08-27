import numpy as np
from collections import Counter
from typing import Tuple
import cv2  # OpenCV for optional preprocessing (blur, etc.)


# DOES NOT WORK YET, BETTER DELETE THIS FUNCTION
# def find_grid_offset(image: np.ndarray, scale: int, threshold: float = 5.0) -> Tuple[int, int]:
#     """
#     Determine the (x_offset, y_offset) of a scaled and cropped binary image.
    
#     Args:
#         image (np.ndarray): 2D grayscale image (values 0–255), already loaded.
#         scale (int): Integer scaling factor used on the original image.
#         threshold (float): Std deviation threshold to detect uniform patches.
    
#     Returns:
#         (int, int): (x_offset, y_offset) in [0, scale-1] range.
#     """
#     if image.ndim != 2:
#         raise ValueError("Image must be grayscale (2D).")

#     H, W = image.shape
#     offset_counts = Counter()

#     for i in range(H - scale + 1):
#         for j in range(W - scale + 1):
#             patch = image[i:i+scale, j:j+scale]
#             if patch.std() < threshold:
#                 dy = i % scale
#                 dx = j % scale
#                 offset_counts[(dx, dy)] += 1

#     if not offset_counts:
#         raise ValueError("No uniform blocks found — consider adjusting threshold or verifying input.")

#     # Most common offset is assumed to be the correct grid alignment
#     (x_offset, y_offset), _ = offset_counts.most_common(1)[0]
#     return x_offset, y_offset

def average_grid_cells(image: np.ndarray, scale: int, x_offset: int, y_offset: int, use_central=False) -> np.ndarray:
    """
    Average all pixels in each cell of the grid based on the scale and offsets.

    Args:
        image (np.ndarray): 2D grayscale image (values 0–255), already loaded.
        scale (int): Integer scaling factor for the grid.
        x_offset (int): Offset of the grid in the x-direction.
        y_offset (int): Offset of the grid in the y-direction.

    Returns:
        np.ndarray: Image with averaged grid cells.
    """
    if image.ndim != 2:
        raise ValueError("Image must be grayscale (2D).")

    H, W = image.shape
    averaged_image = np.zeros_like(image, dtype=np.float32)+1

    for i in range(-y_offset, H, scale):
        for j in range(-x_offset, W, scale):
            cell = image[max(i, 0):min(i+scale, H), max(j, 0):min(j+scale, W)]
            if use_central and cell.size == scale**2:
                # Use the central pixel of the cell if the cell is fully in the frame
                if cell.size > 0:
                    avg_value = cell[cell.shape[0] // 2, cell.shape[1] // 2]
            else:
                avg_value = np.mean(cell)
            averaged_image[max(i, 0):min(i+scale, H), max(j, 0):min(j+scale, W)] = avg_value

    return averaged_image



def get_grid_block_offsets(
    block_id: int,
    x_scale: int = 3,
    y_scale: int = 3,
    block_stride_w: int = 64,
    block_stride_h: int = 64,
    block_w: int = 128,
    block_h: int = 128,
    post_scale_w: int = 684,
    post_scale_h: int = 684,
) -> Tuple[int, int]:
    """
    Calculate the offsets (x_offset, y_offset) for a block based on its ID and scaling parameters.

    Args:
        block_id (int): ID of the block.
        x_scale (int): Scaling factor for width.
        y_scale (int): Scaling factor for height.
        block_stride_w (int): Stride width for blocks.
        block_stride_h (int): Stride height for blocks.
        block_w (int): Width of the block.
        block_h (int): Height of the block.
        post_scale_w (int): Post-scaling width of the block.
        post_scale_h (int): Post-scaling height of the block.

    Returns:
        Tuple[int, int]: (x_offset, y_offset) offsets for the block.
    """
    blocks_per_row = (post_scale_w - block_w) // block_stride_w + 1
    x = (block_id % blocks_per_row) * block_stride_w
    y = (block_id // blocks_per_row) * block_stride_h

    return x % x_scale, y % y_scale