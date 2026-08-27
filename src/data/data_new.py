import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import cv2
from collections import defaultdict
import time

def imgray(img_3c: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_3c, cv2.COLOR_BGR2GRAY) if len(img_3c.shape) == 3 and img_3c.shape[2] == 3 else img_3c

def minmax(
    img: np.ndarray,
    vmin: int | None = None,
    vmax: int | None = None,
    dtype: type = np.float32,
) -> np.ndarray:
    vmin = vmin or img.min()
    vmax = vmax or img.max()
    if vmax==vmin:
        print("Warning: vmax equals vmin in minmax normalization. Returning zeros array.")
        exit(1)
    return ((img - vmin) / (vmax - vmin)).astype(dtype)

def normalize_img(img: np.ndarray) -> np.ndarray:
    return (minmax(imgray(img)) * 255).astype(np.uint8).astype(np.float32)/255.0

class CDPImage:
    """
    A flexible image container that supports multiple loading modes:
    - image_file: Load image from file on demand
    - block_files: Load blocks from separate files on demand  
    - image_and_block_files: Can load either image or blocks from files
    - loaded_image: Image pre-loaded in memory
    - loaded_blocks: All blocks pre-loaded in memory
    """
    
    def __init__(self, 
                 height: int, 
                 width: int,
                 crinfo: str,
                 uid: str,
                 shot: str,
                 image_file_path: Optional[str] = None,
                 block_file_paths: Optional[str] = None,  # Format string with {block_id} placeholder
                 image_settings: Optional[Dict] = None,
                 block_settings: Optional[Dict] = None,
                 mode: str = "image_file"):
        
        self.height = height
        self.width = width
        self.crinfo = crinfo
        self.uid = uid
        self.shot = shot
        self.image_file_path = image_file_path
        self.block_file_paths = block_file_paths
        
        # Default settings 
        # TODO: This is not how it supposed to be as we have a conflict of two inputs
        self.image_settings = image_settings or {
            'h': height, 'w': width, 
            'loading_resize_annealing': False,
            'loading_do_minmax_normalisation': True
        }
        self.block_settings = block_settings or {
            'block_h': 128, 'block_w': 128,
            'stride_h': 64, 'stride_w': 64
        }
        
        # Memory storage
        self.image: Optional[np.ndarray] = None
        self.blocks: Optional[Dict[int, np.ndarray]] = None
        
        # Set initial mode
        self.mode = mode
        self._validate_mode()
    
    def _validate_mode(self):
        """Validate that the current mode is supported with available data"""
        if self.mode == "image_file" and not self.image_file_path:
            raise ValueError("image_file mode requires image_file_path")
        elif self.mode == "block_files" and not self.block_file_paths:
            raise ValueError("block_files mode requires block_file_paths")
        elif self.mode == "image_and_block_files" and not (self.image_file_path and self.block_file_paths):
            raise ValueError("image_and_block_files mode requires both image_file_path and block_file_paths")
    
    def get_h(self) -> int:
        return self.height
    
    def get_w(self) -> int:
        return self.width
    
    def get_n_blocks(self) -> int:
        """Calculate number of blocks based on image dimensions and block settings"""
        block_h, block_w = self.block_settings['block_h'], self.block_settings['block_w']
        stride_h, stride_w = self.block_settings['stride_h'], self.block_settings['stride_w']
        
        n_blocks_h = (self.height - block_h) // stride_h + 1
        n_blocks_w = (self.width - block_w) // stride_w + 1
        return n_blocks_h * n_blocks_w
    
    def _load_image(self) -> np.ndarray:
        """Load image from file and apply settings to match simple_loading.py exactly"""
        if not self.image_file_path or not os.path.exists(self.image_file_path):
            raise FileNotFoundError(f"Image file not found: {self.image_file_path}")
        
        image = cv2.imread(self.image_file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_file_path}")
        
        # Convert to float and normalize to [0, 1] exactly like simple_loading.py
        image = image.astype("float32") / 255.0
        
        # Resize if needed using the same method as simple_loading.py
        target_h, target_w = self.image_settings.get('h', self.height), self.image_settings.get('w', self.width)
        if image.shape != (target_h, target_w):
            # Map interpolation method to OpenCV constant (default to nearest like simple_loading)
            interpolation_methods = {
                "nearest": cv2.INTER_NEAREST,
                "linear": cv2.INTER_LINEAR,
                "cubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4
            }
            interp_method = self.image_settings.get('interpolation', 'nearest')
            interp = interpolation_methods.get(interp_method, cv2.INTER_NEAREST)
            
            image = cv2.resize(image, (target_w, target_h), interpolation=interp)
        
        return image
    
    def _load_block(self, block_id: int) -> np.ndarray:
        """Load a specific block from file"""
        if not self.block_file_paths:
            raise ValueError("No block file paths available")
        
        block_path = self.block_file_paths.format(block_id=block_id)
        if not os.path.exists(block_path):
            raise FileNotFoundError(f"Block file not found: {block_path}")
        
        block = cv2.imread(block_path, cv2.IMREAD_GRAYSCALE)
        if block is None:
            raise ValueError(f"Could not load block from {block_path}")
        
        # Convert to float and normalize exactly like simple_loading.py
        return block.astype("float32") / 255.0
    
    def _load_all_blocks(self) -> Dict[int, np.ndarray]:
        """Load all blocks from files"""
        blocks = {}
        n_blocks = self.get_n_blocks()
        
        for block_id in range(n_blocks):
            blocks[block_id] = self._load_block(block_id)
        
        return blocks
    
    def _extract_block_from_image(self, block_id: int, image: np.ndarray) -> np.ndarray:
        """Extract a block from the full image using same logic as simple_loading.py"""
        block_h, block_w = self.block_settings['block_h'], self.block_settings['block_w']
        stride_h, stride_w = self.block_settings['stride_h'], self.block_settings['stride_w']
        
        img_height, img_width = image.shape[:2]
        
        # Calculate number of blocks in each dimension (same as simple_loading.py)
        num_blocks_y = max(1, (img_height - block_h) // stride_h + 1)
        num_blocks_x = max(1, (img_width - block_w) // stride_w + 1)
        
        # Convert block index to row, column coordinates
        block_y = block_id // num_blocks_x
        block_x = block_id % num_blocks_x
        
        # Calculate pixel coordinates
        y_start = block_y * stride_h
        x_start = block_x * stride_w
        y_end = min(y_start + block_h, img_height)
        x_end = min(x_start + block_w, img_width)
        
        # Extract the block and apply normalize_img like simple_loading.py
        block = image[y_start:y_end, x_start:x_end].copy()
        return normalize_img(block)
    
    def get_image(self) -> np.ndarray:
        """Get the full image based on current mode"""
        if self.mode == "loaded_image":
            if self.image is None:
                raise ValueError("Image not loaded in memory")
            return self.image
        elif self.mode in ["image_file", "image_and_block_files"]:
            return self._load_image()
        elif self.mode in ["block_files", "loaded_blocks"]:
            raise ValueError(f"Cannot load full image in mode {self.mode}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_block(self, block_id: int, expand_channels=False) -> np.ndarray:
        """Get a specific block based on current mode"""
        if self.mode == "loaded_blocks":
            if self.blocks is None or block_id not in self.blocks:
                raise ValueError(f"Block {block_id} not loaded in memory")
            block = self.blocks[block_id]
        elif self.mode in ["block_files", "image_and_block_files"]:
            block = self._load_block(block_id)
        elif self.mode in ["image_file", "loaded_image"]:
            image = self.get_image()
            block = self._extract_block_from_image(block_id, image)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        if expand_channels and len(block.shape) == 2:
            # Make it of size [1, H, W]
            block = block[np.newaxis, :, :]
        return block
    
    def get_all_blocks(self) -> List[np.ndarray]:
        """Get all blocks as a list, similar to load_image(..., block="all")"""
        if self.mode == "loaded_blocks":
            if self.blocks is None:
                raise ValueError("Blocks not loaded in memory")
            # Return blocks in order
            n_blocks = self.get_n_blocks()
            return [self.blocks[i] for i in range(n_blocks) if i in self.blocks]
        elif self.mode in ["block_files", "image_and_block_files"]:
            # Load all blocks from files
            blocks = []
            n_blocks = self.get_n_blocks()
            for block_id in range(n_blocks):
                try:
                    blocks.append(self._load_block(block_id))
                except:
                    # Skip missing blocks
                    continue
            return blocks
        elif self.mode in ["image_file", "loaded_image"]:
            # Extract all blocks from the full image
            image = self.get_image()
            blocks = []
            n_blocks = self.get_n_blocks()
            for block_id in range(n_blocks):
                blocks.append(self._extract_block_from_image(block_id, image))
            return blocks
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def to_mode(self, new_mode: str):
        """Switch between different loading modes"""
        if new_mode == self.mode:
            return
        
        # Load data if switching to memory-intensive modes
        if new_mode == "loaded_image":
            if self.mode in ["image_file", "image_and_block_files"]:
                self.image = self._load_image()
            else:
                raise ValueError(f"Cannot load image from mode {self.mode}")
            # Clear blocks to save memory
            self.blocks = None
            
        elif new_mode == "loaded_blocks":
            if self.mode in ["block_files", "image_and_block_files"]:
                self.blocks = self._load_all_blocks()
            elif self.mode in ["image_file", "loaded_image"]:
                image = self.get_image()
                self.blocks = {}
                for block_id in range(self.get_n_blocks()):
                    self.blocks[block_id] = self._extract_block_from_image(block_id, image)
            # Clear image to save memory unless we need both
            if new_mode != "image_and_block_files":
                self.image = None
                
        # Clear memory when switching to file-based modes
        elif new_mode in ["image_file", "block_files", "image_and_block_files"]:
            if new_mode != "image_and_block_files":
                self.image = None
                self.blocks = None
        
        self.mode = new_mode
        self._validate_mode()


class CDPImageDataset:
    """
    Dataset manager for collections of CDPImage objects.
    Handles multiple datasets with different crinfo and file structures.
    """
    
    def __init__(self, 
                 dataset_paths: Union[str, List[str]],
                 dataset_names: Union[str, List[str]],
                 structure: Dict[str, str],
                 image_settings: Optional[Dict] = None,
                 block_settings: Optional[Dict] = None):
        
        # Ensure lists
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
            
        if len(dataset_paths) != len(dataset_names):
            raise ValueError("Number of dataset_paths must match dataset_names")
        
        self.dataset_paths = dataset_paths
        self.dataset_names = dataset_names
        self.structure = structure
        self.image_settings = image_settings or {}
        self.block_settings = block_settings or {}
        
        # Storage for CDPImage objects
        self.images: Dict[str, Dict[str, Dict[str, CDPImage]]] = {}  # crinfo -> uid -> shot -> CDPImage
        self.available_shots: Dict[str, Dict[str, set]] = {}  # crinfo -> uid -> set of shots
        
        start_time = time.time()
        self._discover_images()
        elapsed = time.time() - start_time
        print(f"[CDPImageDataset] _discover_images() took {elapsed:.2f} seconds.")
        
        self._validate_datasets()
    
    def _validate_datasets(self):
        """Validate that all datasets have at least one UID"""
        empty_datasets = []
        for i, dataset_name in enumerate(self.dataset_names):
            if not self.available_shots.get(dataset_name) or len(self.available_shots[dataset_name]) == 0:
                empty_datasets.append(f"{dataset_name} (path: {self.dataset_paths[i]})")
        
        if empty_datasets:
            raise ValueError(f"The following datasets contain no UIDs: {empty_datasets}. "
                           f"Please check that the dataset paths exist and contain valid data.")
    
    def _discover_images(self):
        """Discover all available images in the dataset paths"""
        for dataset_path, dataset_name in zip(self.dataset_paths, self.dataset_names):
            self.images[dataset_name] = {}
            self.available_shots[dataset_name] = defaultdict(set)
            
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
            # Get structure for this dataset
            if 'datasets' in self.structure:
                dataset_structure = self.structure['datasets'].get(dataset_name)
            else:
                dataset_structure = self.structure.get('type')
            
            # Discover UIDs and shots based on structure
            self._scan_dataset_structure(dataset_path, dataset_name, dataset_structure)
            print("Discovered N=", len(self.images[dataset_name]), "images in dataset:", dataset_name)
    
    def _scan_dataset_structure(self, dataset_path: Path, dataset_name: str, dataset_structure: str):
        """Scan dataset structure to find UIDs and shots"""
        
        if dataset_structure == "template":
            # Structure: dataset_path/{uid}.tiff
            for tiff_file in dataset_path.glob("*.tiff"):
                uid = tiff_file.stem
                shot = "template"  # Special shot name for templates
                
                self.available_shots[dataset_name][uid].add(shot)
                
                if uid not in self.images[dataset_name]:
                    self.images[dataset_name][uid] = {}
                    
                self._create_cdp_image(dataset_name, uid, shot, dataset_path, tiff_file)
                
        elif dataset_structure == "default":
            # Structure: dataset_path/{uid}/{shot}.tiff
            for uid_dir in dataset_path.iterdir():
                if uid_dir.is_dir():
                    uid = uid_dir.name
                    self.images[dataset_name][uid] = {}
                    
                    for tiff_file in uid_dir.glob("*.tiff"):
                        shot = tiff_file.stem
                        self.available_shots[dataset_name][uid].add(shot)
                        
                        # Create CDPImage object
                        self._create_cdp_image(dataset_name, uid, shot, uid_dir, tiff_file)
                        
        elif dataset_structure == "generated":
            # Structure: dataset_path/{uid}/block_{block_id}.tiff
            for uid_dir in dataset_path.iterdir():
                if uid_dir.is_dir():
                    uid = uid_dir.name
                    shot = "generated"  # Special shot name for generated blocks
                    
                    # Check if there are block files
                    block_files = list(uid_dir.glob("block_*.tiff"))
                    if block_files:
                        self.available_shots[dataset_name][uid].add(shot)
                        
                        if uid not in self.images[dataset_name]:
                            self.images[dataset_name][uid] = {}
                            
                        self._create_cdp_image(dataset_name, uid, shot, uid_dir)
        else:
            raise ValueError(f"Unknown dataset structure: {dataset_structure}")
    
    def _parse_filename(self, filename: str, pattern: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse filename to extract uid and shot based on pattern"""
        # Simplified parser - you may need to enhance this
        try:
            if '_{shot}' in pattern and '_{uid}' in pattern:
                parts = filename.split('_')
                if len(parts) >= 2:
                    return parts[0], parts[1].split('.')[0]  # Remove extension
        except:
            pass
        return None, None
    
    def _create_cdp_image(self, dataset_name: str, uid: str, shot: str, 
                         base_path: Path, image_file: Optional[Path] = None):
        """Create a CDPImage object for the given parameters"""
        
        # Get dataset structure
        if 'datasets' in self.structure:
            dataset_structure = self.structure['datasets'].get(dataset_name)
        else:
            dataset_structure = self.structure.get('type')
        
        # Determine file paths based on structure and image_file
        if image_file:
            image_file_path = str(image_file)
        else:
            # Default naming for different structures
            if dataset_structure == "template":
                image_file_path = str(base_path / f"{uid}.tiff")
            elif dataset_structure == "default":
                image_file_path = str(base_path / f"{shot}.tiff")
            elif dataset_structure == "generated":
                image_file_path = None  # Generated structure uses blocks, not full images
            else:
                image_file_path = str(base_path / f"{uid}_{shot}_image.png")  # Generic fallback
        
        # Block files pattern based on structure
        if dataset_structure == "generated":
            block_file_paths = str(base_path / "block_{block_id}.tiff")
        else:
            block_file_paths = str(base_path / f"{uid}_{shot}_block_{{block_id}}.png")
        
        # Determine mode based on available files and structure
        if dataset_structure == "generated":
            mode = "block_files"
            image_file_path = None
        else:
            has_image = Path(image_file_path).exists() if image_file_path else False
            has_blocks = any(Path(block_file_paths.format(block_id=i)).exists() for i in range(10))  # Check first 10
            
            if has_image and has_blocks:
                mode = "image_and_block_files"
            elif has_image:
                mode = "image_file"
                block_file_paths = None
            elif has_blocks:
                mode = "block_files"
                image_file_path = None
            else:
                print(f"Warning: No valid files found for {dataset_name}/{uid}/{shot}")
                return
        
        # Get image dimensions (you may need to customize this)
        height = self.structure.get('height', 684)  # Default from simple_loading.py
        width = self.structure.get('width', 684)
        
        cdp_image = CDPImage(
            height=height,
            width=width,
            crinfo=dataset_name,
            uid=uid,
            shot=shot,
            image_file_path=image_file_path,
            block_file_paths=block_file_paths,
            image_settings=self.image_settings,
            block_settings=self.block_settings,
            mode=mode
        )
        
        self.images[dataset_name][uid][shot] = cdp_image
    
    def _is_wildcard_shot(self, shot: str) -> bool:
        """Check if a shot is a wildcard (generated or template)"""
        return shot in ["generated", "template"]
    
    def filter_common_uids(self):
        """Filter to keep only UIDs that are available in ALL datasets"""
        if len(self.dataset_names) <= 1:
            return
        
        # Find common UIDs across all datasets
        common_uids = set.intersection(*[set(self.available_shots[name].keys()) for name in self.dataset_names])
        # Remove UIDs not in common from each dataset
        for dataset_name in self.dataset_names:
            uids_to_remove = []
            for uid in list(self.available_shots[dataset_name].keys()):
                if uid not in common_uids:
                    uids_to_remove.append(uid)
            
            # Remove UIDs not in common
            for uid in uids_to_remove:
                if uid in self.images[dataset_name]:
                    del self.images[dataset_name][uid]
                if uid in self.available_shots[dataset_name]:
                    del self.available_shots[dataset_name][uid]
    
    def filter_common_shots(self):
        """Filter to keep only shots that are available for each UID in ALL datasets.
        Wildcard shots (generated, template) act as universal wildcards that can represent any shot."""
        if len(self.dataset_names) <= 1:
            return
        
        # For each UID that exists in all datasets, find common shots
        for dataset_name in self.dataset_names:
            uids_to_remove = []
            for uid in list(self.available_shots[dataset_name].keys()):
                # Check if this UID exists in all other datasets
                uid_exists_in_all = all(uid in self.available_shots[other_dataset] 
                                      for other_dataset in self.dataset_names)
                
                if uid_exists_in_all:
                    # Get shots for this UID across all datasets, separating wildcards and non-wildcards
                    shots_per_dataset = []
                    all_wildcard_shots = set()
                    all_non_wildcard_shots = set()
                    
                    for other_dataset in self.dataset_names:
                        if uid in self.available_shots[other_dataset]:
                            dataset_shots = self.available_shots[other_dataset][uid]
                            non_wildcard = {shot for shot in dataset_shots if not self._is_wildcard_shot(shot)}
                            wildcards = {shot for shot in dataset_shots if self._is_wildcard_shot(shot)}
                            
                            shots_per_dataset.append({
                                'non_wildcard': non_wildcard,
                                'wildcard': wildcards,
                                'all': dataset_shots
                            })
                            all_wildcard_shots.update(wildcards)
                            all_non_wildcard_shots.update(non_wildcard)
                    
                    # Find shots that are "available" in all datasets
                    # A shot is available in a dataset if:
                    # 1. It exists as non-wildcard in that dataset, OR
                    # 2. The dataset has any wildcard shot (which can represent any shot)
                    shots_to_keep = set()
                    
                    # Check each unique non-wildcard shot to see if it can be "covered" by all datasets
                    for shot in all_non_wildcard_shots:
                        available_in_all = True
                        for dataset_shots in shots_per_dataset:
                            # Shot is available if it's explicitly in non_wildcard OR there are wildcards that can represent it
                            if shot not in dataset_shots['non_wildcard'] and not dataset_shots['wildcard']:
                                available_in_all = False
                                break
                        
                        if available_in_all:
                            shots_to_keep.add(shot)
                    
                    # Always keep all wildcard shots (they represent universal availability)
                    shots_to_keep.update(all_wildcard_shots)
                    
                    # Remove shots not in the keep set
                    shots_to_remove = [shot for shot in list(self.available_shots[dataset_name][uid]) 
                                     if shot not in shots_to_keep]
                    for shot in shots_to_remove:
                        if uid in self.images[dataset_name] and shot in self.images[dataset_name][uid]:
                            del self.images[dataset_name][uid][shot]
                        self.available_shots[dataset_name][uid].discard(shot)
                    
                    # Remove UID if no shots remain
                    if not self.available_shots[dataset_name][uid]:
                        uids_to_remove.append(uid)
                else:
                    # UID doesn't exist in all datasets, mark for removal
                    uids_to_remove.append(uid)
            
            # Remove UIDs with no remaining shots or that don't exist in all datasets
            for uid in uids_to_remove:
                if uid in self.images[dataset_name]:
                    del self.images[dataset_name][uid]
                if uid in self.available_shots[dataset_name]:
                    del self.available_shots[dataset_name][uid]

    def filter_common_uids_shots(self):
        """Filter to keep only UIDs and shots that are available in ALL datasets.
        This is a convenience method that combines filter_common_uids() and filter_common_shots().
        Wildcard shots (generated, template) are always kept and not considered for common shots."""
        self.filter_common_uids()
        self.filter_common_shots()

    def filter_shots(self, shots_to_keep: Union[str, List[str], int, List[int]], 
                     include_dataset_names: Optional[List[str]] = None,
                     exclude_dataset_names: Optional[List[str]] = None):
        """Filter to keep only specific shots across all datasets.
        Wildcard shots (generated, template) always pass this filter.
        
        Args:
            shots_to_keep: Single shot or list of shots to keep. Can be strings or integers.
                          Integers will be converted to zero-padded 4-digit strings (e.g., 1 -> "0001").
        """
        # Ensure list and convert to strings
        if not isinstance(shots_to_keep, list):
            shots_to_keep = [shots_to_keep]
        
        # Convert all to strings, handle integer formatting
        shots_to_keep_str = []
        for shot in shots_to_keep:
            if isinstance(shot, int):
                shots_to_keep_str.append(f"{shot:04d}")
            else:
                shots_to_keep_str.append(str(shot))
        
        shots_to_keep_set = set(shots_to_keep_str)
        
        if include_dataset_names is None:
            include_dataset_names = self.dataset_names
        
        if exclude_dataset_names is not None:
            # Exclude specified datasets
            include_dataset_names = [name for name in self.dataset_names if name not in exclude_dataset_names] 
        
        # Filter shots in each dataset
        for dataset_name in include_dataset_names:
            uids_to_remove = []
            for uid in list(self.available_shots[dataset_name].keys()):
                shots_to_remove = []
                
                # Find shots to remove for this UID
                for shot in list(self.available_shots[dataset_name][uid]):
                    # Keep shot if it's in the keep list OR if it's a wildcard
                    if shot not in shots_to_keep_set and not self._is_wildcard_shot(shot):
                        shots_to_remove.append(shot)
                
                # Remove filtered shots
                for shot in shots_to_remove:
                    if uid in self.images[dataset_name] and shot in self.images[dataset_name][uid]:
                        del self.images[dataset_name][uid][shot]
                    self.available_shots[dataset_name][uid].discard(shot)
                
                # Remove UID if no shots remain
                if not self.available_shots[dataset_name][uid]:
                    uids_to_remove.append(uid)
            
            # Remove UIDs with no remaining shots
            for uid in uids_to_remove:
                if uid in self.images[dataset_name]:
                    del self.images[dataset_name][uid]
                if uid in self.available_shots[dataset_name]:
                    del self.available_shots[dataset_name][uid]

    def filter_uids_by_range(self, uid_range: List[int]):
        """Filter UIDs by numeric range, keeping only UIDs within [min_uid, max_uid]"""
        min_uid, max_uid = uid_range
        
        for dataset_name in self.dataset_names:
            uids_to_remove = []
            for uid in list(self.available_shots[dataset_name].keys()):
                try:
                    uid_num = int(uid)
                    if not (min_uid <= uid_num <= max_uid):
                        uids_to_remove.append(uid)
                except ValueError:
                    # Remove UIDs that are not numeric
                    uids_to_remove.append(uid)
            
            # Remove filtered UIDs
            for uid in uids_to_remove:
                if uid in self.images[dataset_name]:
                    del self.images[dataset_name][uid]
                if uid in self.available_shots[dataset_name]:
                    del self.available_shots[dataset_name][uid]

    def get_CDPimage(self, crinfo: str, uid: Union[str, int], shot: str = None) -> CDPImage:
        """Get a specific CDPImage object"""
        if isinstance(uid, int):
            uid = f"{uid:06d}"
        try:
            if shot is None:
                shot = next(iter(self.available_shots[crinfo][uid]))
            return self.images[crinfo][uid][shot]
        except KeyError:
            raise KeyError(f"Image not found: {crinfo}/{uid}/{shot}")
    
    def get_all_CDPimages(self) -> List[CDPImage]:
        """Get all CDPImage objects in the dataset"""
        all_images = []
        for crinfo in self.images:
            for uid in self.images[crinfo]:
                for shot in self.images[crinfo][uid]:
                    all_images.append(self.images[crinfo][uid][shot])
        return all_images
    
    def get_images_by_crinfo(self, crinfo: str) -> List[CDPImage]:
        """Get all images for a specific crinfo"""
        images = []
        if crinfo in self.images:
            for uid in self.images[crinfo]:
                for shot in self.images[crinfo][uid]:
                    images.append(self.images[crinfo][uid][shot])
        return images
    
    def get_crinfos(self) -> List[str]:
        """Get list of all crinfo values"""
        return list(self.images.keys())
    
    def get_uids(self, crinfo: Optional[str] = None) -> List[str]:
        """Get list of UIDs, optionally filtered by crinfo"""
        if crinfo:
            return list(self.available_shots.get(crinfo, {}).keys())
        else:
            all_uids = set()
            for dataset_shots in self.available_shots.values():
                all_uids.update(dataset_shots.keys())
            return list(all_uids)
    
    def get_n_uids(self, crinfo: Optional[str] = None) -> int:
        """Get the number of UIDs. If crinfo is provided, returns count for that crinfo only.
        Args:
            crinfo: Dataset name (crinfo) to count UIDs for. If None, counts unique UIDs across all datasets.
        Returns:
            Integer count of UIDs.
        """
        if crinfo:
            return len(self.available_shots.get(crinfo, {}))
        # count unique across all datasets
        all_uids = set()
        for dataset_shots in self.available_shots.values():
            all_uids.update(dataset_shots.keys())
        return len(all_uids)

    def get_shots(self, crinfo: str, uid: str) -> List[str]:
        """Get list of shots for a specific crinfo and uid"""
        return list(self.available_shots.get(crinfo, {}).get(uid, set()))
    
    def print_info(self):
        # Print for available datasets the range of UIDs and shots and the total number of shots available
        for dataset_name in self.dataset_names:
            print(f"\nDataset: {dataset_name}")
            print("-" * 30)
            print("Min UID:", min(self.available_shots[dataset_name].keys(), key=int, default="N/A"))
            print("Max UID:", max(self.available_shots[dataset_name].keys(), key=int, default="N/A"))
            print("Total UIDs:", len(self.available_shots[dataset_name]))
            # check if the shots are numbers or strings
            if len(self.available_shots[dataset_name]) > 0:
                if next(iter(next(iter(self.available_shots[dataset_name].values())))).isdigit():
                    print("min shotid:", min(set.union(*list(self.available_shots[dataset_name].values())), default="N/A"))
                    print("max shotid:", max(set.union(*list(self.available_shots[dataset_name].values())), default="N/A"))
                else:
                    print("shots of the first UID:", list(self.available_shots[dataset_name].values())[0]) 
                print("Total shots:", sum(len(shots) for shots in self.available_shots[dataset_name].values()))
    
    def print_all_available_shots(self):
        """Print all available shots for each UID across all datasets"""
        print("Available shots by dataset and UID:")
        print("=" * 50)
        
        for dataset_name in self.dataset_names:
            print(f"\nDataset: {dataset_name}")
            print("-" * 30)
            
            if dataset_name not in self.available_shots or not self.available_shots[dataset_name]:
                print("  No UIDs available")
                continue
            
            # Sort UIDs numerically if possible, otherwise alphabetically
            uids = list(self.available_shots[dataset_name].keys())
            try:
                uids.sort(key=int)
            except ValueError:
                uids.sort()
            
            for uid in uids:
                shots = list(self.available_shots[dataset_name][uid])
                # Sort shots numerically if possible, otherwise alphabetically
                try:
                    shots.sort(key=int)
                except ValueError:
                    shots.sort()
                
                shots_str = ", ".join(shots)
                print(f"  UID {uid}: [{shots_str}] ({len(shots)} shots)")
        
        print("\n" + "=" * 50)
        total_images = len(self.get_all_CDPimages())
        print(f"Total images across all datasets: {total_images}")
    
    def set_mode_all(self, mode: str):
        """Set the mode for all CDPImage objects in the dataset"""
        for image in self.get_all_CDPimages():
            image.to_mode(mode)
    
    def __len__(self) -> int:
        """Return total number of images in the dataset"""
        return len(self.get_all_images())
    
    def __repr__(self) -> str:
        return f"CDPImageDataset(datasets={len(self.dataset_names)}, total_images={len(self)})"
    
    @classmethod
    def from_image_datasets(cls, 
                           dataset_names: Union[str, List[str]], 
                           dataset_base_path: str,
                           reference_base_path: Optional[str] = None,
                           structures: Optional[Dict[str, str]] = None,
                           image_settings: Optional[Dict] = None,
                           block_settings: Optional[Dict] = None):
        """
        Create CDPImageDataset from dataset names using image_datasets configuration.
        
        Args:
            dataset_names: Single dataset name or list of dataset names
            dataset_base_path: Base path for datasets
            reference_base_path: Optional separate base path for reference datasets
            structures: Optional override for dataset structures
            image_settings: Optional image loading settings
            block_settings: Optional block extraction settings
        """
        from src.utils.default_datasets import image_datasets
        
        # Ensure lists
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        
        dataset_paths = []
        structures_dict = {}
        
        #print(dataset_names)
        #exit(1)
        
        for dataset_name in dataset_names:
            
            # Use reference_base_path for template datasets, otherwise use dataset_base_path
            if dataset_name in image_datasets:
                dataset_info = image_datasets[dataset_name]
                rel_path = dataset_info["rel_path"]
                structure_type = structures.get(dataset_name, dataset_info["structure"]) #structures canoverride default structure
                
                # Use reference_base_path for template structures
                if structure_type == "template" and reference_base_path:
                    base_path = reference_base_path
                else:
                    base_path = dataset_base_path
                    
                full_path = str(Path(base_path) / rel_path.lstrip('/'))
            elif dataset_name[:-3] in image_datasets and dataset_name[-3:]=="DUP":
                dataset_info = image_datasets[dataset_name[:-3]]
                rel_path = dataset_info["rel_path"]
                structure_type = structures.get(dataset_name, dataset_info["structure"]) #structures can override default
                # Use reference_base_path for template structures
                if structure_type == "template" and reference_base_path:
                    base_path = reference_base_path
                else:
                    base_path = dataset_base_path
                full_path = str(Path(base_path) / rel_path.lstrip('/'))
            else:
                # For datasets not in image_datasets, use provided structure
                structure_type = structures.get(dataset_name) if structures else None
                if not structure_type:
                    raise ValueError(f"Dataset '{dataset_name}' not found in image_datasets and no structure provided")
                full_path = str(Path(dataset_base_path) / dataset_name)
            
            dataset_paths.append(full_path)
            structures_dict[dataset_name] = structure_type
        
        # Convert structures to the format expected by CDPImageDataset
        structure = {'datasets': structures_dict}
        
        return cls(dataset_paths, dataset_names, structure, image_settings, block_settings)

def simple_load_and_plot(uid: str, dataset_name: str, shot: str = None, block_id: int = 0,
                        dataset_base_path: str = "data/wifs2024dataset/wifs2024dataset",
                        save_path: str = "loaded_image.png"):
    """
    Simple function to load an image using CDPImage classes and plot it as grayscale.
    
    Args:
        uid: Image UID
        dataset_name: Name of the dataset (e.g., "o55iP12w")
        shot: Shot number (string or int), if None uses template mode
        block_id: Block ID to load (0-based index)
        dataset_base_path: Base path to dataset
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    from src.utils.default_datasets import image_datasets
    
    try:
        # Get dataset info
        dataset_info = image_datasets[dataset_name]
        rel_path = dataset_info["rel_path"]
        structure_type = dataset_info["structure"]
        
        # Construct file paths based on structure
        if structure_type == "template":
            image_file_path = str(Path(dataset_base_path) / rel_path / f"{uid}.tiff")
            crinfo = f"{dataset_name}_template"
            shot_name = "template"
        else:  # default structure
            if shot is None:
                raise ValueError("Shot required for default structure")
            if isinstance(shot, int):
                shot_name = f"{shot:04d}"
            else:
                shot_name = shot
            image_file_path = str(Path(dataset_base_path) / rel_path.lstrip('/') / uid / f"{shot_name}.tiff")
            crinfo = f"{dataset_name}_default"
        
        # Create CDPImage with matching settings
        cdp_image = CDPImage(
            height=684, width=684,  # Default target size from simple_loading
            crinfo=crinfo,
            uid=uid,
            shot=shot_name,
            image_file_path=image_file_path,
            image_settings={'h': 684, 'w': 684, 'interpolation': 'nearest'},
            block_settings={'block_h': 128, 'block_w': 128, 'stride_h': 64, 'stride_w': 64},
            mode="image_file"
        )
        
        # Load the specific block
        block = cdp_image.get_block(block_id)
        
        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.imshow(block, cmap='gray')
        plt.title(f"Dataset: {dataset_name}, UID: {uid}, Shot: {shot_name}, Block: {block_id}\nShape: {block.shape}")
        plt.axis('off')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Successfully loaded and plotted block")
        print(f"  Dataset: {dataset_name}")
        print(f"  UID: {uid}")
        print(f"  Shot: {shot_name}")
        print(f"  Block ID: {block_id}")
        print(f"  Block shape: {block.shape}")
        print(f"  Plot saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load and plot block: {e}")
        return False

def compare_loading_methods(uid: str, dataset_name: str, shot: str = None, block_id: int = 0,
                          dataset_base_path: str = "data/wifs2024dataset/wifs2024dataset"):
    """
    Compare image loading between simple_loading.py and CDPImageDataset to ensure identical results.
    """
    from src.data.simple_loading import load_image
    from src.utils.default_datasets import image_datasets
    
    # Load using simple_loading.py
    try:
        simple_img = load_image(uid, dataset_name, shot=shot, dataset_base_path=dataset_base_path)
        simple_block = load_image(uid, dataset_name, shot=shot, dataset_base_path=dataset_base_path, block=block_id)
        print(f"✓ simple_loading.py: Image shape {simple_img.shape}, Block shape {simple_block.shape}")
    except Exception as e:
        print(f"✗ simple_loading.py failed: {e}")
        return False
    
    # Load using CDPImageDataset
    try:
        # Get dataset info to construct CDPImageDataset
        dataset_info = image_datasets[dataset_name]
        rel_path = dataset_info["rel_path"]
        structure_type = dataset_info["structure"]
        
        # Construct file paths based on structure
        if structure_type == "template":
            image_file_path = str(Path(dataset_base_path) / rel_path / f"{uid}.tiff")
            crinfo = f"{dataset_name}_template"
        else:  # default structure
            if shot is None:
                raise ValueError("Shot required for default structure")
            if isinstance(shot, int):
                shot = f"{shot:04d}"
            image_file_path = str(Path(dataset_base_path) / rel_path.lstrip('/') / uid / f"{shot}.tiff")
            crinfo = f"{dataset_name}_default"
        
        # Create CDPImage with matching settings
        cdp_image = CDPImage(
            height=684, width=684,  # Default target size from simple_loading
            crinfo=crinfo,
            uid=uid,
            shot=shot or "template",
            image_file_path=image_file_path,
            image_settings={'h': 684, 'w': 684, 'interpolation': 'nearest'},
            block_settings={'block_h': 128, 'block_w': 128, 'stride_h': 64, 'stride_w': 64},
            mode="image_file"
        )
        
        cdp_img = cdp_image.get_image()
        cdp_block = cdp_image.get_block(block_id)
        print(f"✓ CDPImageDataset: Image shape {cdp_img.shape}, Block shape {cdp_block.shape}")
        
    except Exception as e:
        print(f"✗ CDPImageDataset failed: {e}")
        return False
    
    # Compare arrays
    img_equal = np.allclose(simple_img, cdp_img, rtol=1e-6, atol=1e-6)
    block_equal = np.allclose(simple_block, cdp_block, rtol=1e-6, atol=1e-6)
    
    print(f"Image arrays equal: {img_equal}")
    print(f"Block arrays equal: {block_equal}")
    
    if not img_equal:
        print(f"Image difference - Mean: {np.mean(np.abs(simple_img - cdp_img))}, Max: {np.max(np.abs(simple_img - cdp_img))}")
    if not block_equal:
        print(f"Block difference - Mean: {np.mean(np.abs(simple_block - cdp_block))}, Max: {np.max(np.abs(simple_block - cdp_block))}")
    
    return img_equal and block_equal

if __name__ == "__main__":
    # Test simple load and plot function
    print("Testing simple_load_and_plot function:")
    simple_success = simple_load_and_plot(
        uid="000261",
        dataset_name="o55iPXSo", 
        shot="0001",
        block_id=0,
        save_path="test_image_plot.png"
    )
    print(f"Simple load and plot success: {simple_success}")
    print()
    
    # Test comparison
    print("Testing compare_loading_methods function:")
    comparison_success = compare_loading_methods(
        uid="000263",
        dataset_name="o55iP12w", 
        shot="0001",
        block_id=0
    )
    print(f"Loading methods match: {comparison_success}")
