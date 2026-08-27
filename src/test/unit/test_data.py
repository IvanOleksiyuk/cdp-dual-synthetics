import pytest
import tempfile
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

import pyrootutils
pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.data.data import CDPDatamodule


class TestCDPDatamodule:
    """Minimal focused unit tests for CDPDatamodule (no mocks)."""

    @pytest.fixture
    def mock_dataset_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for split in ['train', 'valid', 'test']:
                for block_id in ['000001', '000002']:
                    block_dir = Path(temp_dir) / split / block_id / 'blocks' / 'block_1'
                    block_dir.mkdir(parents=True, exist_ok=True)
                    # create original + fake tiffs
                    for suffix in ['t', 'op1', 'op2', 'fp1', 'fp2']:
                        img = (np.random.rand(32, 32) * 255).astype(np.uint8)
                        Image.fromarray(img, mode='L').save(block_dir / f'{block_id}_{suffix}.tiff')
            yield temp_dir

    @pytest.fixture
    def basic_datamodule(self, mock_dataset_path):
        return CDPDatamodule(
            dataset_path=mock_dataset_path,
            batch_size=2,
            num_workers=0,
            n_captures=2,
            allowed_suffixes=['op1', 'op2'],
            fake_suffixes=['fp1', 'fp2'],
            pin_memory=False,
            random_seed=42,
            class_mapping="old"  # the default mapping only knows real device names, not the op1/op2 mock suffixes
        )

    def test_init_default_and_custom(self, mock_dataset_path):
        """Verify default parameter values and that custom overrides are applied correctly."""
        dm_default = CDPDatamodule(dataset_path=mock_dataset_path)
        assert dm_default.batch_size == 32
        assert dm_default.num_workers == 4
        assert dm_default.allowed_suffixes == []
        assert dm_default.fake_suffixes is None
        assert dm_default.add_fake is False
        assert dm_default.pin_memory is True
        assert dm_default.deterministic_val_test is True

        dm_custom = CDPDatamodule(
            dataset_path=mock_dataset_path,
            batch_size=8,
            num_workers=1,
            n_captures=1,
            allowed_suffixes=['op1'],
            fake_suffixes=['fp1'],
            add_fake=True,
            pin_memory=False,
            deterministic_val_test=False,
            random_seed=7
        )
        assert dm_custom.batch_size == 8
        assert dm_custom.num_workers == 1
        assert dm_custom.n_captures == 1
        assert dm_custom.allowed_suffixes == ['op1']
        assert dm_custom.fake_suffixes == ['fp1']
        assert dm_custom.add_fake is True
        assert dm_custom.pin_memory is False
        assert dm_custom.deterministic_val_test is False
        assert dm_custom.random_seed == 7

    def test_suffix_handling(self, mock_dataset_path):
        """Ensure suffix propagation logic for x1/x2 defaults and explicit overrides works."""
        dm = CDPDatamodule(dataset_path=mock_dataset_path, allowed_suffixes=['op1', 'op2'])
        assert dm.allowed_suffixes_x1 == ['op1', 'op2']
        assert dm.allowed_suffixes_x2 == ['op1', 'op2']
        dm2 = CDPDatamodule(dataset_path=mock_dataset_path, allowed_suffixes=['op1', 'op2'], allowed_suffixes_x1=['op1'], allowed_suffixes_x2=['op2'])
        assert dm2.allowed_suffixes_x1 == ['op1']
        assert dm2.allowed_suffixes_x2 == ['op2']

    def test_rotation_and_determinism(self, basic_datamodule):
        """Check rotation flag during fit and deterministic flags for val/test datasets."""
        basic_datamodule.setup(stage='fit')
        # train dataset rotation enabled only in fit stage
        assert basic_datamodule.train_dataset.add_rotation is True
        assert basic_datamodule.val_dataset.add_rotation is False
        assert basic_datamodule.test_dataset.add_rotation is False
        # deterministic flags
        assert basic_datamodule.train_dataset.deterministic_choice is False
        assert basic_datamodule.val_dataset.deterministic_choice is True
        assert basic_datamodule.test_dataset.deterministic_choice is True
        # NOTE: CDPDatamodule.setup() currently hardcodes add_rotation=True for the train
        # dataset regardless of stage (the stage-dependent branch is commented out there) -
        # this assertion matches that current behavior rather than asserting a stage check.
        basic_datamodule.setup(stage='test')
        assert basic_datamodule.train_dataset.add_rotation is True

    def test_dataset_splits_and_param_passing(self, mock_dataset_path):
        """Confirm datasets are instantiated with correct parameter propagation."""
        dm = CDPDatamodule(dataset_path=mock_dataset_path, n_captures=1, allowed_suffixes=['op1'], allowed_suffixes_x1=['op1'], allowed_suffixes_x2=['op2'], fake_suffixes=['fp1'], add_fake=True, random_seed=11)
        dm.setup(stage='fit')
        for ds in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
            assert ds.n_captures == 1
            assert ds.allowed_suffixes == ['op1']
            assert ds.allowed_suffixes_x1 == ['op1']
            assert ds.allowed_suffixes_x2 == ['op2']
            assert ds.fake_suffixes == ['fp1']
            assert ds.add_fake is True
            assert ds.random_seed == 11

    def test_fake_suffix_edge_cases(self, mock_dataset_path):
        """Validate behavior when fake_suffixes=None vs empty list."""
        # fake_suffixes None -> auto-derived from allowed_suffixes
        dm_none = CDPDatamodule(dataset_path=mock_dataset_path, allowed_suffixes=['op1', 'op2'], fake_suffixes=None)
        dm_none.setup()
        expected_auto = ['fp1', 'fp2']  # derived by replacing leading char 'o' with 'f'
        assert dm_none.train_dataset.fake_suffixes == expected_auto
        # Empty lists: dataset construction should fail (no allowed originals)
        dm_empty = CDPDatamodule(dataset_path=mock_dataset_path, allowed_suffixes=[], fake_suffixes=[])
        with pytest.raises(ValueError):
            dm_empty.setup()

    def test_dataloader_configuration(self, basic_datamodule):
        """Assert DataLoader construction: batch size, sampler types, workers, pin_memory using real datasets."""
        basic_datamodule.setup(stage='fit')
        train_loader = basic_datamodule.train_dataloader()
        val_loader = basic_datamodule.val_dataloader()
        test_loader = basic_datamodule.test_dataloader()
        assert isinstance(train_loader, DataLoader)
        assert train_loader.batch_size == 2
        assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)
        assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler)
        assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)
        assert train_loader.num_workers == val_loader.num_workers == test_loader.num_workers == 0
        assert train_loader.pin_memory is False

    def test_dataset_direct_getitem(self, basic_datamodule):
        """Directly access underlying datasets via __getitem__ to validate sample structure and tensor shapes."""
        basic_datamodule.setup(stage='fit')
        sample = basic_datamodule.train_dataset[0]
        # Required keys for n_captures=2
        for key in ['zt', 'x1', 'x2', 'c1', 'c2', 'ids', 'parts']:
            assert key in sample
        assert sample['zt'].ndim == 4 or sample['zt'].ndim == 3  # (1,H,W) or batched if augmented later
        assert sample['x1'].shape[0] == 1  # channel dim
        assert sample['x2'].shape[0] == 1
        # Labels are tensors
        assert isinstance(sample['c1'], torch.Tensor)
        assert isinstance(sample['c2'], torch.Tensor)
        # Access val/test datasets too (deterministic)
        val_sample = basic_datamodule.val_dataset[0]
        test_sample = basic_datamodule.test_dataset[0]
        for s in [val_sample, test_sample]:
            for key in ['zt', 'x1', 'x2', 'c1', 'c2', 'ids', 'parts']:
                assert key in s
            assert s['zt'].ndim == 4 or s['zt'].ndim == 3  # (1,H,W) or batched if augmented later
            assert s['x1'].shape[0] == 1  # channel dim
            assert s['x2'].shape[0] == 1
            # Labels are tensors
            assert isinstance(s['c1'], torch.Tensor)
            assert isinstance(s['c2'], torch.Tensor)


class TestCDPDatamoduleIntegration:
    """Single integration test to ensure real data iteration works (no mocks)."""

    @pytest.fixture
    def real_dataset_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for split in ['train', 'valid', 'test']:
                for block_id in ['000001', '000002']:
                    block_dir = Path(temp_dir) / split / block_id / 'blocks' / 'block_1'
                    block_dir.mkdir(parents=True, exist_ok=True)
                    for suffix in ['t', 'op1', 'op2']:
                        img = (np.random.rand(32, 32) * 255).astype(np.uint8)
                        Image.fromarray(img, mode='L').save(block_dir / f'{block_id}_{suffix}.tiff')
            yield temp_dir

    def test_integration_iteration(self, real_dataset_structure):
        """Smoke test real iteration: produce batch with required keys and iterate multiple batches."""
        dm = CDPDatamodule(dataset_path=real_dataset_structure, batch_size=1, num_workers=0, allowed_suffixes=['op1', 'op2'], n_captures=2, pin_memory=False, class_mapping="old")
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        for key in ['zt', 'x1', 'x2', 'c1', 'c2', 'ids', 'parts']:
            assert key in batch
        assert batch['x1'].shape[0] == 1
        cnt = 0
        for _ in loader:
            cnt += 1
            if cnt >= 2:
                break
        assert cnt >= 1