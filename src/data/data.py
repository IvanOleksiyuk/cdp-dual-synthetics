import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import cv2
import torchvision.transforms as transforms
import numpy as np

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.utils.default_datasets import get_default_class_mapping


class CDPDataset(Dataset):
    def __init__(self, dataset_path: str, 
                 split: str, 
                 n_captures: int = 2, 
                 apply_augmentations: bool = True, 
                 allowed_suffixes: List[str] = None,
                 allowed_suffixes_x1: List[str] = None, 
                 allowed_suffixes_x2: List[str] = None,
                 exclude_selected_x1_from_x2: bool = True, 
                 fake_suffixes: List[str] = None, 
                 add_fake: bool = False,
                 add_rotation: bool = False,
                 deterministic_choice: bool = False,
                 random_seed: int = 42,
                 class_mapping = None):
        """
        Initialize the CDPDataset.

        Args:
            dataset_path (str): Path to the dataset (e.g., train/valid/test folder).
            split (str): Dataset split to load (e.g., "train", "valid", "test").
            n_captures (int): Number of captures to load (1 or 2).
            apply_augmentations (bool): Whether to apply random augmentations.
            allowed_suffixes (List[str]): List of allowed suffixes for selecting original captures (e.g., ["op1", "op2"]).
            fake_suffixes (List[str]): List of fake suffixes corresponding to the allowed_suffixes (e.g., ["fp1", "fp2"]).
            add_fake (bool): Whether to load and include fake captures in the results.
            deterministic_choice (bool): If True, pre-generate deterministic choices for file selection.
            random_seed (int): Random seed for deterministic choice generation.
        """
        self.n_captures = n_captures
        self.dataset_path = Path(dataset_path) / split
        self.allowed_suffixes = allowed_suffixes if allowed_suffixes is not None else []
                # All the stuff abou allowed suffixes
        if allowed_suffixes_x1 is None:
            self.allowed_suffixes_x1 = self.allowed_suffixes
        else:
            self.allowed_suffixes_x1 = allowed_suffixes_x1
        if allowed_suffixes_x2 is None:
            self.allowed_suffixes_x2 = self.allowed_suffixes
        else:
            self.allowed_suffixes_x2 = allowed_suffixes_x2
        self.exclude_selected_x1_from_x2 = exclude_selected_x1_from_x2
        self.deterministic_choice = deterministic_choice
        self.random_seed = random_seed
        
        self.fake_suffixes = fake_suffixes if fake_suffixes is not None else [f"f{s[1:]}" for s in self.allowed_suffixes]
        self.block_folders = self._filter_block_folders()  # Ensure allowed_suffixes is initialized before this call
        if self.block_folders is None or len(self.block_folders) == 0:
            #print("Data/CDPDataset/__init__/self.block_folders", self.block_folders)
            print(f"Allowed suffixes: {self.allowed_suffixes}")
            raise ValueError(f"No valid block folders found in {self.dataset_path} with allowed suffixes: {self.allowed_suffixes}")
        self.apply_augmentations = apply_augmentations
        self.add_fake = add_fake
        self.add_rotation = add_rotation
        
        # Pre-generate deterministic choices if requested
        self.predetermined_choices = None
        if self.deterministic_choice:
            self.predetermined_choices = self._generate_predetermined_choices()
            
        if len(self.allowed_suffixes) != len(self.fake_suffixes):
            raise ValueError("The number of allowed_suffixes and fake_suffixes must match.")
        
        if class_mapping is None:
            self.class_mapping = get_default_class_mapping()
        elif class_mapping == "old":
            self.class_mapping = self._get_old_class_mapping()
        

    def _find_file_with_suffix(self, folder: Path, suffix: str) -> Path | None:
        """
        Find a file with the given suffix in the specified folder.

        Args:
            folder (Path): The folder to search in.
            suffix (str): The suffix to look for.

        Returns:
            Path | None: The found file path or None if not found.
        """
        for file in folder.glob(f"*{suffix}.tiff"):
            if file.is_file():
                return file
        return None

    def _count_present_suffixes(self, folder: Path, suffixes: List[str]) -> int:
        count = 0
        for suffix in suffixes:
            if self._find_file_with_suffix(folder, suffix):
                count += 1
        return count

    def _filter_block_folders(self) -> List[Path]:
        """
        Filter block folders to include only those that contain files with allowed suffixes.

        Returns:
            List[Path]: List of block folders that contain at least one file with an allowed suffix.
        """
        all_block_folders = list((Path(self.dataset_path)).glob("*/blocks/block_*"))
        #print("Data/CDPDataset/_filter_block_folders/self.dataset_path:", self.dataset_path)
        #print("Data/CDPDataset/_filter_block_folders/all_block_folders:", all_block_folders)
        filtered_folders = []
        for block_folder in all_block_folders:
            files = list(block_folder.glob("*.tiff"))
            x1_pass = False
            x2_pass = False
            if self.allowed_suffixes_x1==self.allowed_suffixes_x2 and self.n_captures == 2:
                # When x1 and x2 are drawn from the same pool
                # Count for each folder how many correct sufixes we find
                count = self._count_present_suffixes(block_folder, self.allowed_suffixes_x1)
                if count >= 2:
                    x1_pass = True
                    x2_pass = True
            else:
                # When x1 and x2 are drawn from different pools (assume non-overlaping)
                if any(file.stem.endswith(suffix) for file in files for suffix in self.allowed_suffixes_x1):
                    x1_pass= True
                if self.n_captures > 1:
                    if any(file.stem.endswith(suffix) for file in files for suffix in self.allowed_suffixes_x2):
                        x2_pass = True
            
            # Decide to keep or not based on the passes 
            if self.n_captures > 1:
                if x1_pass and x2_pass:
                    filtered_folders.append(block_folder)
            else:
                if x1_pass:
                    filtered_folders.append(block_folder)
        return filtered_folders

    def _generate_predetermined_choices(self) -> List[Dict]:
        """
        Generate predetermined file choices for each block folder to ensure deterministic behavior.
        
        Returns:
            List[Dict]: List of predetermined choices for each block folder.
        """
        # Set seed for reproducible choices
        random_state = random.getstate()
        random.seed(self.random_seed)
        
        predetermined_choices = []
        
        for block_folder in self.block_folders:
            files = list(block_folder.glob("*.tiff"))
            
            # Filter class files based on allowed suffixes
            class_files_x1 = [file for file in files if not file.stem.endswith("_t") and any(file.stem.endswith(suffix) for suffix in self.allowed_suffixes_x1)]
            
            choice_dict = {}
            
            if self.n_captures == 1:
                # Pre-select x1 file
                choice_dict['x1_file'] = random.choice(class_files_x1)
            else:
                # Pre-select x1 and x2 files
                class_files_x2 = [file for file in files if not file.stem.endswith("_t") and any(file.stem.endswith(suffix) for suffix in self.allowed_suffixes_x2)]
                
                x1_file = random.choice(class_files_x1)
                choice_dict['x1_file'] = x1_file
                
                if self.exclude_selected_x1_from_x2:
                    available_x2_files = [file for file in class_files_x2 if file != x1_file]
                    if available_x2_files:
                        choice_dict['x2_file'] = random.choice(available_x2_files)
                    else:
                        # Fallback to any x2 file if no alternatives
                        choice_dict['x2_file'] = random.choice(class_files_x2)
                else:
                    choice_dict['x2_file'] = random.choice(class_files_x2)
            
            predetermined_choices.append(choice_dict)
        
        # Restore original random state
        random.setstate(random_state)
        
        return predetermined_choices

    def _get_old_class_mapping(self) -> Dict[str, int]:
        """
        Create a mapping of class labels to discrete identifiers.

        Returns:
            Dict[str, int]: Mapping of class labels (e.g., "t", "p1", "s2") to integers.
        """
        class_labels = set()
        for block_folder in self.block_folders:
            class_labels.update([file.stem.split("_")[-1] for file in block_folder.glob("*.tiff")])
        return {label: idx for idx, label in enumerate(sorted(class_labels))}
        
    def __len__(self) -> int:
        return len(self.block_folders)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.

        Args:
            idx (int): Index of the block folder.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing "zt", "x1", "c1", "ids", "parts" (for n_captures=1)
                                     or "zt", "x1", "x2", "c1", "c2", "ids", "parts" (for n_captures=2).
        """
        block_folder = self.block_folders[idx]
        files = list(block_folder.glob("*.tiff"))

        # Extract block ID and part ID from the folder structure
        block_id = block_folder.parent.parent.name  # Correctly extract the ID (e.g., "000190")
        part_id = block_folder.name  # e.g., "block_1"

        # Load the template block
        zt_file = next(file for file in files if file.stem.endswith("_t"))
        zt = self._load_image(zt_file)

        # Filter class files based on allowed suffixes
        class_files_x1 = [file for file in files if not file.stem.endswith("_t") and any(file.stem.endswith(suffix) for suffix in self.allowed_suffixes_x1)]
        if self.n_captures > 1:
            class_files_x2 = [file for file in files if not file.stem.endswith("_t") and any(file.stem.endswith(suffix) for suffix in self.allowed_suffixes_x2)]
        
        if not class_files_x1:
            raise ValueError(f"No valid captures found in {block_folder} with allowed suffixes: {self.allowed_suffixes}")

        # Filter fake class files based on fake suffixes
        # for file in files: # FOR DEBUGGING
        #     print(file.stem[7]) # FOR DEBUGGING
        # exit() # FOR DEBUGGING
        fake_files = [file for file in files if not file.stem.endswith("_t") and any(file.stem.endswith(suffix) for suffix in self.fake_suffixes)]

        if self.n_captures == 1:
            # Select one original capture
            if self.deterministic_choice and self.predetermined_choices:
                x1_file = self.predetermined_choices[idx]['x1_file']
            else:
                x1_file = random.choice(class_files_x1)
                
            x1_label = x1_file.stem.split("_")[-1]
            x1 = self._load_image(x1_file)
            c1 = self.class_mapping[x1_label]

            # Apply augmentations if specified
            x1 = self._apply_min_max_scaling(x1)[0] #[0] to escape the tuple
            if self.add_rotation:
                x1, zt = self._apply_rotation(x1, zt)
            
            result = {
                "zt": zt,
                "x1": x1,
                "c1": torch.tensor(c1, dtype=torch.float32),
                "ids": block_id,
                "parts": part_id,
            }

            # Add fake capture if required
            if self.add_fake:
                f1_file = next((file for file in fake_files if file.stem.endswith(x1_label)), None)
                if f1_file is None:
                    raise ValueError(f"No matching fake capture found for original class {x1_label} in {block_folder}")
                f1 = self._load_image(f1_file)
                result["f1"] = f1

            return result

        # For n_captures == 2, select two original captures
        if self.deterministic_choice and self.predetermined_choices:
            x1_file = self.predetermined_choices[idx]['x1_file']
            x2_file = self.predetermined_choices[idx]['x2_file']
        else:
            x1_file = random.choice(class_files_x1)
            if self.exclude_selected_x1_from_x2:
                # Exclude the selected x1_file from x2_file selection
                class_files_x2 = [file for file in class_files_x2 if file!=x1_file]            
                if len(class_files_x2) == 0:
                    raise ValueError(f"No valid captures found for x2 in {block_folder} after excluding x1: {x1_file}")
                    
            x2_file = random.choice(class_files_x2)
        x1_label = x1_file.stem.split("_")[-1]
        x2_label = x2_file.stem.split("_")[-1]
        x1 = self._load_image(x1_file)
        x2 = self._load_image(x2_file)
        c1 = self.class_mapping[x1_label]
        c2 = self.class_mapping[x2_label]
        
        x1, x2 = self._apply_min_max_scaling(x1, x2)
        if self.add_rotation:
            x1, x2, zt = self._apply_rotation(x1, x2, zt)
        
        result = {
            "zt": zt,
            "x1": x1,
            "c1": torch.tensor(c1, dtype=torch.float32),
            "x2": x2,
            "c2": torch.tensor(c2, dtype=torch.float32),
            "ids": block_id,
            "parts": part_id,
        }

        # Add fake captures if required
        if self.add_fake:
            f1_file = next((file for file in fake_files if file.stem.endswith(x1_label)), None)
            f2_file = next((file for file in fake_files if file.stem.endswith(x2_label)), None)
            if f1_file is None or f2_file is None:
                raise ValueError(f"No matching fake captures found for original classes {x1_label} and {x2_label} in {block_folder}")
            f1 = self._load_image(f1_file)
            f2 = self._load_image(f2_file)
            result["f1"] = f1
            result["f2"] = f2

        return result

    def _load_image(self, file_path: Path) -> torch.Tensor:
        """
        Load an image from a file in grayscale mode, normalize it, and convert it to a PyTorch tensor.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            torch.Tensor: The normalized grayscale image as a tensor.
        """
        # Load the image in grayscale mode
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        if img.dtype == "uint16":  # Convert uint16 to float32 for compatibility
            img = img.astype("float32")
        else:
            img = img.astype("float32")  # Ensure the image is float32

        # Normalize the image to the range [0, 1]
        img /= 255.0

        # Convert to PyTorch tensor and add channel dimension
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img

    def _apply_augmentations(self, *images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply the same random augmentations to all input images.

        Args:
            *images (torch.Tensor): Images to be augmented.

        Returns:
            Tuple[torch.Tensor, ...]: Augmented images.
        """
        # Define random augmentation transformations
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.RandomRotation(degrees=(0, 0)),
                transforms.RandomRotation(degrees=(90, 90)),
                transforms.RandomRotation(degrees=(180, 180)),
                transforms.RandomRotation(degrees=(270, 270)),
            ]),
        ])

        # Generate a random seed to ensure the same augmentations are applied to all images
        seed = np.random.randint(0, 10000)

        augmented_images = []
        for img in images:
            torch.manual_seed(seed)  # Set the seed for reproducibility
            augmented_images.append(augmentation(img))

        return tuple(augmented_images)

    def _apply_min_max_scaling(self, *images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply min-max scaling to all input images.

        Args:
            *images (torch.Tensor): Images to be scaled.

        Returns:
            Tuple[torch.Tensor, ...]: Scaled images.
        """
        scaled_images = []
        for img in images:
            # Scale the image to the range [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = torch.zeros_like(img)
            scaled_images.append(img)
        return tuple(scaled_images)

    def _apply_rotation(self, *images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Apply the same random rotation and flipping to all input images.

        Args:
            *images (torch.Tensor): Images to be rotated and flipped.

        Returns:
            Tuple[torch.Tensor, ...]: Rotated and flipped images.
        """
        # Randomly select a rotation angle from [0, 90, 180, 270]
        rotation_angle = random.choice([0, 90, 180, 270])

        # Randomly decide whether to flip horizontally or vertically (50% chance for each)
        flip_horizontal = random.choice([True, False])
        flip_vertical = random.choice([True, False])

        transformed_images = []
        for img in images:
            # Rotate the image
            if rotation_angle != 0:
                img = torch.rot90(img, k=rotation_angle // 90, dims=(1, 2))

            # Flip the image horizontally if selected
            if flip_horizontal:
                img = torch.flip(img, dims=[2])

            # Flip the image vertically if selected
            if flip_vertical:
                img = torch.flip(img, dims=[1])

            transformed_images.append(img)

        return tuple(transformed_images)


class CDPDatamodule(LightningDataModule):
    def __init__(self, 
                 dataset_path: str, 
                 batch_size: int = 32, 
                 num_workers: int = 4, 
                 n_captures: int = 2, 
                 allowed_suffixes_x1: List[str] = None, 
                 allowed_suffixes_x2: List[str] = None,
                 allowed_suffixes: List[str] = None, 
                 fake_suffixes: List[str] = None, 
                 add_fake: bool = False,
                 pin_memory: bool = True,
                 deterministic_val_test: bool = True,
                 random_seed: int = 42,
                 class_mapping=None):
        """
        Initialize the CDPDatamodule.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size for the dataloaders.
            num_workers (int): Number of workers for the dataloaders.
            n_captures (int): Number of captures to load (1 or 2).
            allowed_suffixes (List[str]): List of allowed suffixes for selecting original captures (e.g., ["op1", "op2"]).
            fake_suffixes (List[str]): List of fake suffixes corresponding to the allowed_suffixes (e.g., ["fp1", "fp2"]).
            add_fake (bool): Whether to load and include fake captures in the results.
            deterministic_val_test (bool): If True, validation and test datasets will use deterministic file choices.
            random_seed (int): Random seed for deterministic choice generation.
            class_mapping: Passed through to CDPDataset (None for the default device mapping, "old" to derive
                one dynamically from whatever suffixes are actually present in the dataset).
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_captures = n_captures
        self.allowed_suffixes = allowed_suffixes if allowed_suffixes is not None else []
        self.allowed_suffixes_x1 = allowed_suffixes_x1 if allowed_suffixes_x1 is not None else self.allowed_suffixes
        self.allowed_suffixes_x2 = allowed_suffixes_x2 if allowed_suffixes_x2 is not None else self.allowed_suffixes
        self.fake_suffixes = fake_suffixes
        self.add_fake = add_fake
        self.pin_memory = pin_memory
        self.deterministic_val_test = deterministic_val_test
        self.random_seed = random_seed
        self.class_mapping = class_mapping

    def setup(self, stage: str = None):
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str): Stage of the setup (e.g., "fit", "test").
        """
        # Enable augmentations only for the training dataset
        self.train_dataset = CDPDataset(
            self.dataset_path, 
            "train", 
            n_captures=self.n_captures, 
            apply_augmentations=False, 
            allowed_suffixes=self.allowed_suffixes, 
            allowed_suffixes_x1=self.allowed_suffixes_x1,
            allowed_suffixes_x2=self.allowed_suffixes_x2,
            fake_suffixes=self.fake_suffixes, 
            add_fake=self.add_fake,
            add_rotation=True, #if stage == "fit" else False,
            deterministic_choice=False,  # Keep training randomized
            random_seed=self.random_seed,
            class_mapping=self.class_mapping,
        )
        self.val_dataset = CDPDataset(
            self.dataset_path, 
            "valid", 
            n_captures=self.n_captures, 
            apply_augmentations=False, 
            allowed_suffixes=self.allowed_suffixes, 
            allowed_suffixes_x1=self.allowed_suffixes_x1,
            allowed_suffixes_x2=self.allowed_suffixes_x2,
            fake_suffixes=self.fake_suffixes, 
            add_fake=self.add_fake,
            deterministic_choice=self.deterministic_val_test,  # Make validation deterministic
            random_seed=self.random_seed,
            class_mapping=self.class_mapping,
        )
        self.test_dataset = CDPDataset(
            self.dataset_path, 
            "test", 
            n_captures=self.n_captures, 
            apply_augmentations=False, 
            allowed_suffixes=self.allowed_suffixes,
            allowed_suffixes_x1=self.allowed_suffixes_x1,
            allowed_suffixes_x2=self.allowed_suffixes_x2, 
            fake_suffixes=self.fake_suffixes, 
            add_fake=self.add_fake,
            deterministic_choice=self.deterministic_val_test,  # Make testing deterministic
            random_seed=self.random_seed,
            class_mapping=self.class_mapping,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)


if __name__ == "__main__":
    # Example usage
    dataset_path = "data/cdp_transit_dataset"
    batch_size = 16

    datamodule = CDPDatamodule(
        dataset_path, 
        num_workers=1, 
        batch_size=batch_size, 
        n_captures=2, 
        allowed_suffixes=["f55Epson"], 
        allowed_suffixes_x2=["f55iP12w"],
        deterministic_val_test=True,  # Make validation/test deterministic
        random_seed=42
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    import matplotlib.pyplot as plt

    for batch in train_loader:
        print("Batch contents:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):  # Check if the value is a tensor
                print(f"{key}: {value.shape}")
            elif isinstance(value, list):  # Handle lists
                print(f"{key}: List of length {len(value)}")
            else:  # Handle other types
                print(f"{key}: {type(value)}")
        # Plot images for each tensor key with 4D shape (B, 1, H, W)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                n = min(4, value.shape[0])  # Show up to 4 images per key
                fig, axs = plt.subplots(1, n, figsize=(4*n, 4))
                fig.suptitle(key)
                for i in range(n):
                    img = value[i, 0].cpu().numpy()
                    axs[i].imshow(img, cmap='gray')
                    axs[i].set_title(f"{key}[{i}]")
                    axs[i].axis('off')
                plt.show()
        break