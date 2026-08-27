from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

from src.data.data_new import CDPImageDataset
from src.utils.default_datasets import get_default_class_mapping  # type: ignore
log = logging.getLogger(__name__)


class NewCDPDataset(Dataset):
    """Wrap a (already filtered) CDPImageDataset into sample dicts expected by tests."""

    def __init__(self,
                 dataset_base_path, 
                 template_data, 
                 original_data_x1, 
                 original_data_x2 = None,
                 uid_range=[],
                 shots_to_keep=1, 
                 class_mapping: Optional[Dict[str, str]] = None,
                 image_settings: Optional[Dict[str, Any]] = None,
                 block_settings = None,
                 preload: bool = True):

        # Normalize block settings
        if block_settings is None:
            block_settings = {}
        elif not isinstance(block_settings, dict):
            block_settings = dict(block_settings)
        
        self.dataset =  CDPImageDataset.from_image_datasets(
            dataset_names = [template_data, original_data_x1] + ([original_data_x2] if original_data_x2 else []),
            dataset_base_path=dataset_base_path,
            structures={},
            image_settings=image_settings,
            block_settings=block_settings)
        
        self.template_data = template_data
        self.original_data_x1 = original_data_x1
        self.original_data_x2 = original_data_x2
        if original_data_x2:
            self.do_add_x2 = True
        else:
            self.do_add_x2 = False
        
        if not uid_range==[]:
            self.dataset.filter_uids_by_range(uid_range)
        self.dataset.filter_common_uids()
        self.dataset.filter_shots(shots_to_keep)
        
        self.total_uids = self.dataset.get_n_uids(self.template_data)
        self.uid_list = self.dataset.get_uids(self.template_data)
        self.blocks_per_image = self.dataset.get_images_by_crinfo(self.template_data)[0].get_n_blocks()
        
        if class_mapping is not None:
            self.class_mapping = class_mapping
        else:
            self.class_mapping = get_default_class_mapping()
        
        if preload:
            print("Starting to preload images...")
            self.dataset.set_mode_all("loaded_image")
            print("finished preloading images")
        
        if len(self.uid_list) == 0:
            raise ValueError("No UIDs found in the dataset after filtering. Check your filters and dataset contents.")
        
        
    def __len__(self):
        return self.total_uids * self.blocks_per_image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        uid_idx = idx // self.blocks_per_image
        block_idx = idx % self.blocks_per_image
        uid = self.uid_list[uid_idx]
        template_image = self.dataset.get_CDPimage(self.template_data, uid)
        original_image_x1 = self.dataset.get_CDPimage(self.original_data_x1, uid)
        if self.do_add_x2:
            original_image_x2 = self.dataset.get_CDPimage(self.original_data_x2, uid)
            return {
                "zt": template_image.get_block(block_idx, expand_channels=True),
                "x1": original_image_x1.get_block(block_idx, expand_channels=True),
                "c1": self.class_mapping.get(self.original_data_x1, "unknown"),
                "x2": original_image_x2.get_block(block_idx, expand_channels=True),
                "c2": self.class_mapping.get(self.original_data_x2, "unknown"),
                "ids": uid,
                "parts": f"block_{block_idx}"
            }
        else:
            return {
                "zt": template_image.get_block(block_idx, expand_channels=True),
                "x1": original_image_x1.get_block(block_idx, expand_channels=True),
                "c1": self.class_mapping.get(self.original_data_x1, "unknown"),
                "ids": uid,
                "parts": f"block_{block_idx}"
            }

class NewCDPDatamodule(LightningDataModule):
    """LightningDataModule to build identical train/val/test DataLoaders from a single
    NewCDPDataset configuration provided via `dataset_config`.

    dataset_config keys (all passed straight to NewCDPDataset):
    - dataset_base_path: str | Path
    - template_data: str
    - original_data_x1: str
    - original_data_x2: Optional[str]
    - uid_range: Optional[List[int]]
    - shots_to_keep: int
    - class_mapping: Optional[Dict[str, str]]
    - image_settings: Optional[Dict[str, Any]]
    - block_settings: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        # DataLoader params (non-dataset related)
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: Optional[bool] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        self.dataset_cfg = copy.deepcopy(dataset_config)
        # Basic validation
        if "dataset_base_path" not in self.dataset_cfg:
            raise ValueError("dataset_config must include 'dataset_base_path'.")

        # Loader settings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._persistent_workers = (
            persistent_workers if persistent_workers is not None else num_workers > 0
        )
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Single dataset instance reused across splits
        self._dataset: Optional[NewCDPDataset] = None
        self.train_dataset: Optional[NewCDPDataset] = None
        self.val_dataset: Optional[NewCDPDataset] = None
        self.test_dataset: Optional[NewCDPDataset] = None

    def _build_dataset(self) -> NewCDPDataset:
        cfg = self.dataset_cfg
        return NewCDPDataset(
            dataset_base_path=cfg["dataset_base_path"],
            template_data=cfg["template_data"],
            original_data_x1=cfg["original_data_x1"],
            original_data_x2=cfg.get("original_data_x2"),
            uid_range=cfg.get("uid_range", []),
            shots_to_keep=cfg.get("shots_to_keep", 1),
            class_mapping=cfg.get("class_mapping"),
            image_settings=cfg.get("image_settings"),
            block_settings=cfg.get("block_settings"),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if self._dataset is None:
            self._dataset = self._build_dataset()
        # Use the same dataset across splits
        self.train_dataset = self._dataset
        self.val_dataset = self._dataset
        self.test_dataset = self._dataset

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule not set up. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule not set up. Call setup('validate') or setup('fit') first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule not set up. Call setup('test') first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self._persistent_workers,
            drop_last=False,
        )
    
    
if __name__ == "__main__":
    # Quick smoke test for dataset
    dataset = NewCDPDataset(
        dataset_base_path="data/wifs2024dataset/wifs2024dataset",
        template_data="tem",
        original_data_x1="o55Epson",
        block_settings={'block_h': 128, 'block_w': 128, 'stride_h': 64, 'stride_w': 64},
    )
    print("Single dataset length:", len(dataset))
    print("First sample keys:", list(dataset[0].keys()))

    # Quick smoke test for datamodule (same dataset for all splits)
    dm = NewCDPDatamodule(
        dataset_config={
            'dataset_base_path': 'data/wifs2024dataset/wifs2024dataset',
            'template_data': 'tem',
            'original_data_x1': 'o55Epson',
            'shots_to_keep': 1,
            'block_settings': {'block_h': 128, 'block_w': 128, 'stride_h': 64, 'stride_w': 64},
        },
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("Train batch keys:", list(batch.keys()))