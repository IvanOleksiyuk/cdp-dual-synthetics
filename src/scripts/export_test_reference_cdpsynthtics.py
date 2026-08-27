# This is a script to export test dataset transformed images using a trained Pix2PixWF model. from cdpsynthetics repository.

############################################################################
import os

# Ensure reproducibility for CUDA operations
# choose one of the two values; :16:8 uses less memory, :4096:8 may be faster
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
# make Python hashing deterministic (affects dict/set iteration order in some cases)
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch as T
T.use_deterministic_algorithms(True)
T.backends.cudnn.deterministic = True
T.backends.cudnn.benchmark = False
############################################################################

from pathlib import Path
import cv2
import logging
import hydra
from omegaconf import DictConfig
import numpy as np
import json
import pytorch_lightning as pl
from tqdm.auto import tqdm

#local imports
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
print(root)
import cdpsynthetics.cdpsynt.pix2pix.model as roma_pix2pix
from src.data.data import CDPDatamodule


log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="l2cond_EpsoniP12iP14_dev")
def main(cfg: DictConfig) -> None:
    if cfg.export.get("generation_seed", None) is not None:
        pl.seed_everything(cfg.export.generation_seed, workers=True)
    else:
        if cfg.seed:
            log.info(f"Setting seed to: {cfg.seed}")
            pl.seed_everything(cfg.seed, workers=True)    

    log.info(f"Loading checkpoint from {cfg.export.checkpoint}")
    device = "cuda" if T.cuda.is_available() else "cpu"
    model_class = roma_pix2pix.Pix2PixWF
    #model = model_class.load_from_checkpoint("data/wifs2024dataset/wifs2024dataset/out/experiment/2nd_iPXS_55/checkpoints/epoch_11_val_auc_ssim_synthetic_0.958.ckpt", map_location=device)
    model = model_class.load_from_checkpoint("data/wifs2024dataset/wifs2024dataset/out/experiment/2nd_iP15uw_55/checkpoints/epoch_05_val_auc_ssim_synthetic_0.999.ckpt", map_location=device)
    model.eval()

    cfg.export.reference_name="2nd_iP15uw_55"

    log.info("Setting up the data module")
    if cfg.data.get("export_datamodule", None) is not None:
        datamodule = hydra.utils.instantiate(cfg.data.export_datamodule)
        datamodule.setup(stage="test")
        log.info(f"Using export datamodule: {cfg.data.export_datamodule._target_}")
    else:
        #exit(1) #This part is deprecated TODO: remove
        datamodule = hydra.utils.instantiate(cfg.data.datamodule)
        datamodule.setup(stage="test")
        log.info(f"Using default datamodule in test regime: {cfg.data.datamodule._target_}")
    test_loader = datamodule.test_dataloader()

    output_dir = Path(cfg.export.export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create an inverse mapping for class IDs to class names
    inverse_class_mapping = {v: k for k, v in datamodule.train_dataset.class_mapping.items()}

    # Determine the specific class the model is dedicated to
    dedicated_class_from_training = None
    if len(cfg.data.datamodule.allowed_suffixes) == 1:
        dedicated_class_from_training = cfg.data.datamodule.allowed_suffixes[0]

    log.info(f"Initial dedicated class based on training: {dedicated_class_from_training}")

    log.info("Processing test dataset")
    with T.no_grad():
        pbar_total = len(test_loader) if hasattr(test_loader, "__len__") else None
        for batch_idx, batch in enumerate(tqdm(test_loader, total=pbar_total, desc="Exporting", unit="batch", dynamic_ncols=True)):
            zt = batch["zt"].to(device)  # Move zt to the same device as the model
            ids = batch["ids"]  # Image IDs
            parts = batch["parts"]  # Block IDs
            reference_name = cfg.export.reference_name  # Use the first allowed suffix as the capture method

            if cfg.export.get("capture_item", None) is None:
                # Export without conditioning
                transformed = model.action_generate(zt)
                for i, (transformed_img, img_id, part) in enumerate(zip(transformed, ids, parts)):
                    img_id_str = str(img_id) if isinstance(img_id, str) else img_id[i]
                    part_str = str(part) if isinstance(part, str) else part[i]
                    output_path = output_dir / f"{reference_name}/{img_id_str}/{part_str}.tiff"

                    transformed_img_np = transformed_img.squeeze().cpu().numpy() * 255.0
                    transformed_img_np = np.clip(transformed_img_np, 0, 255).astype("uint8")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), transformed_img_np)
                    if cfg.export.get("verbose", False):
                        log.info(f"Saved transformed image to {output_path}")

            else:
                # Export for each class
                if hasattr(datamodule, "train_dataset") and datamodule.train_dataset is not None:
                    class_id = datamodule.train_dataset.class_mapping[cfg.export.get("capture_item", None)]
                else:
                    class_id = datamodule.test_dataset.class_mapping[cfg.export.get("capture_item", None)]
                class_name = inverse_class_mapping[class_id]  # Use inverse mapping to get class name
                transformed = model(zt)
                for i, (transformed_img, img_id, part) in enumerate(zip(transformed, ids, parts)):
                    img_id_str = str(img_id) if isinstance(img_id, str) else img_id[i]
                    #print(part)
                    #exit(1)
                    part_str = part if isinstance(part, str) else str(part.item())
                    output_path = output_dir / f"{reference_name}_class_{class_name}/{img_id_str}/{part_str}.tiff"

                    transformed_img_np = transformed_img.squeeze().cpu().numpy() * 255.0
                    transformed_img_np = np.clip(transformed_img_np, 0, 255).astype("uint8")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), transformed_img_np)
                    if cfg.export.get("verbose", False):
                        log.info(f"Saved transformed image for class {class_name} to {output_path}")


    # Collect metadata for the reference folder
    if cfg.export.get("capture_item", None) is None:
        # Save metadata.json in the reference folder
        metadata_path = output_dir / f"{reference_name}/metadata.json"
        metadata = {
            "model_name": cfg.paths.model_name,
            "class_condition_used": None,
            "dedicated_class": dedicated_class_from_training
        }
        if cfg.get("undedicated_name", None):
            metadata["reference_print_label_for_trauncation"] = cfg.undedicated_name
        done_path = output_dir / f"{reference_name}/export.DONE"
    else:
        # Save metadata.json in the reference folder
        metadata_path = output_dir / f"{reference_name}_class_{class_name}/metadata.json"
        metadata = {
            "model_name": cfg.paths.model_name,
            "reference_print_label_for_trauncation": cfg.paths.model_name,
            "class_condition_used": class_name,
            "dedicated_class": class_name, # Override dedicated_class for export_classes
        }
        done_path = output_dir / f"{reference_name}_class_{class_name}/export.DONE"
    # Write metadata to the file
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    log.info(f"Saved metadata to {metadata_path}")
    # Write export.DONE in the reference folder for each class
        
    with open(done_path, "a") as file:
        file.write("DONE")

    log.info(f"Exported transformed test data to {output_dir}")

if __name__ == "__main__":
    main()
