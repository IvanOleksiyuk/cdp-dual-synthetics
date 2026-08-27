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
from tqdm import tqdm
import gc

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
from src.data.data import CDPDatamodule, CDPDataset
from src.model.image_transit import ImageTRANSIT

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="WIP_l2condtransport_EpsoniP12iP14")
def main(cfg: DictConfig) -> None:
    log.info(f"Loading checkpoint from {cfg.export.checkpoint}")
    device = "cuda" if T.cuda.is_available() else "cpu"
    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(cfg.export.checkpoint, map_location=device)
    model.eval()

    log.info("Setting up the data module")

    # Load transport pairs
    transport_pairs = cfg.export.get("transport_pairs", [])
    output_dir = Path(cfg.export.export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_name = cfg.export.get("reference_name", "reference")

    log.info(f"Transport pairs: {transport_pairs}")

    with T.no_grad():
        for from_class, to_class in transport_pairs:
            log.info(f"Generating references: {from_class} -> {to_class}")
            # Create a datamodule for this transport pair, set allowed_suffixes_x2 and allowed_suffixes_x1
            log.info("Setting up the data module")
            if cfg.data.get("export_datamodule", None) is not None:
                datamodule_cfg = cfg.data.export_datamodule.copy()
                datamodule_cfg.dataset_config.original_data_x1 = to_class
                datamodule_cfg.dataset_config.original_data_x2 = from_class
                datamodule = hydra.utils.instantiate(datamodule_cfg)
                datamodule.setup(stage="test")
                test_dataset = datamodule.test_dataset
                test_loader = datamodule.test_dataloader()
                log.info(f"Using export datamodule: {cfg.data.export_datamodule._target_}")
            else:
                #exit(1) #This part is deprecated TODO: remove
                datamodule_cfg = cfg.data.datamodule.copy()
                datamodule_cfg.allowed_suffixes_x2 = [from_class]
                datamodule_cfg.allowed_suffixes_x1 = [to_class]
                datamodule = hydra.utils.instantiate(datamodule_cfg)
                datamodule.setup(stage="test")
                test_dataset = datamodule.test_dataset
                test_loader = datamodule.test_dataloader()
                log.info(f"Using default datamodule in test regime: {cfg.data.datamodule._target_}")

            out_dir = output_dir / f"{reference_name}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for batch in tqdm(test_loader, desc=f"Batches {from_class}"):
                x2 = batch["x2"].to(device)
                c1 = batch["c1"].to(device)
                c2 = batch["c2"].to(device)
                zt = batch["zt"].to(device)  # Not used, but required by the model
                ids = batch["ids"]
                parts = batch["parts"]
                # True transport: encode from x2/from_class, decode to to_class
                generated = model.action_transport(x2, c1=c1, c2=c2, zt=zt)
                generated = generated.cpu().numpy()
                for i in range(x2.shape[0]):
                    img = (generated[i] * 255).squeeze().clip(0, 255).astype(np.uint8)
                    image_id = ids[i]
                    part_id = parts[i]
                    # Save each block in a separate folder for every id
                    block_dir = out_dir / f"{image_id}"
                    block_dir.mkdir(parents=True, exist_ok=True)
                    out_path = block_dir / f"{part_id}.tiff"
                    cv2.imwrite(str(out_path), img)
            # Save metadata.json in the reference folder (once per folder)
            metadata_path = out_dir / "metadata.json"
            metadata = {
                "model_name": cfg.paths.model_name,
                "reference_print_label_for_trauncation": cfg.paths.model_name+"_from_"+from_class,
                "class_condition_from": from_class,
                "class_condition_to": to_class,
                "dedicated_class": to_class
            }
            with open(metadata_path, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=4)
            log.info(f"Saved metadata to {metadata_path}")
            # Write export.DONE in the reference folder for each transport pair
            done_path = out_dir / "export.DONE"
            with open(done_path, "a") as file:
                file.write("DONE")
            # Cleanup datamodule and memory
            del datamodule
            del test_dataset
            del test_loader
            gc.collect()
            if device == "cuda":
                T.cuda.empty_cache()
            log.info(f"Done: {from_class} -> {to_class}")

    # Remove global export.DONE, now handled per reference folder
    log.info(f"Exported transported references to {output_dir}")

if __name__ == "__main__":
    main()
