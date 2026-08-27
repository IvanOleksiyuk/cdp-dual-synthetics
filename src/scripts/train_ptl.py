import os
import torch as T

import logging
import hydra
import pytorch_lightning as pl
import time
from pathlib import Path
from omegaconf import DictConfig, open_dict

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

from src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config

log = logging.getLogger(__name__)

def parse_tags(cfg: DictConfig) -> list:
    """
    Parse tags from the configuration.
    """
    tags = []
    if hasattr(cfg, "tags"):
        if isinstance(cfg.tags, list):
            tags.extend(cfg.tags)
        elif isinstance(cfg.tags, str):
            tags.append(cfg.tags)
    return tags

@hydra.main(version_base=None, config_path="../configs", config_name="model_debug")
def main(cfg: DictConfig) -> None:

    if cfg.get("deterministic", True):
        # Ensure reproducibility for CUDA operations
        # choose one of the two values; :16:8 uses less memory, :4096:8 may be faster
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        # make Python hashing deterministic (affects dict/set iteration order in some cases)
        os.environ.setdefault("PYTHONHASHSEED", "0")
        T.use_deterministic_algorithms(True)
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False

    # Ensure the full_path directory exists

    full_path = Path(cfg.paths.full_path)
    if not full_path.exists():
        log.info(f"Creating directory: {full_path}")
        full_path.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize wandb if key is available
        if cfg.get("wandb_key", False):
            wandb_key = cfg.wandb_key
        elif os.getenv("WANDB_API_KEY"):
            wandb_key = os.getenv("WANDB_API_KEY")
        else:
            wandb_key = open(cfg.paths.wandbkey, "r").read()

        if wandb_key=="skip":
            wandb = None 
            log.info("W&B key is set to 'skip'. Skipping W&B initialization.")
        else:
            if wandb_key in ["", "null", "None"]:
                raise ValueError("W&B key is empty. Please provide a valid W&B API key.")

            import wandb
            wandb.login(key=wandb_key)
            run_id = wandb.util.generate_id()

            wandb.init(project=cfg.project_name, id=run_id, name=cfg.network_name, resume="allow", tags=parse_tags(cfg))
            with open(cfg.paths.full_path + "/wandb_id.txt", "w") as f:
                f.write(run_id)
    except (ImportError, FileNotFoundError, KeyError) as e:
        wandb = None
        log.warning(f"Could not initialize wandb: {e}. Proceeding without wandb.")

    if hasattr(cfg, "loggers") and hasattr(cfg.loggers, "wandb"):
        if wandb is not None and wandb.run is not None and hasattr(cfg.loggers.wandb, "struct") and not cfg.loggers.wandb.struct:
            cfg.loggers.wandb.id = wandb.run.id
        if wandb is None:
            # No real W&B run: still keep the logger object (some callbacks, e.g.
            # LearningRateMonitor, require at least one logger to exist), but make PL's
            # lazily-triggered wandb.init() fully offline/no-op instead of hitting the network.
            os.environ.setdefault("WANDB_MODE", "disabled")
        # "struct" is our own metadata field, not a real WandbLogger/wandb.init kwarg -
        # instantiate_collection() would otherwise forward it straight into wandb.init().
        with open_dict(cfg.loggers.wandb):
            cfg.loggers.wandb.pop("struct", None)

    log.info("Setting up full job config")
    if cfg.full_resume:
        cfg = reload_original_config(cfg)
    else:
        # if a folder with checkpoints already exists we have to delete it with all the files
        if os.path.exists(cfg.paths.full_path + "checkpoints"):
            log.info(f"Deleting folder: {cfg.paths.full_path + 'checkpoints'}")
            os.system(f"rm -rf {cfg.paths.full_path + 'checkpoints'}")
    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)

    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model)
    log.info(model)

    # # Check if adversarial training is enabled
    # if hasattr(model, "adversarial") and model.adversarial:
    #     log.info("Adversarial training detected. Enabling manual optimization in the Trainer.")
    #     cfg.trainer.automatic_optimization = False  # Ensure manual optimization is enabled for adversarial training

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    accelerator = "gpu" if T.cuda.is_available() else "cpu"
    devices = T.cuda.device_count() if T.cuda.is_available() else 1
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers, accelerator=accelerator, devices=devices)

    log.info("Starting training!")
    start_time = time.time()

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    if wandb is not None and loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds")
    formatted_time = f"Execution Time: {elapsed_time:.2f} seconds\n"

    # Write the elapsed time to a text file
    with open(full_path / "execution_time.txt", "a") as file:
        file.write(formatted_time)

    with open(full_path / "training.DONE", "a") as file:
        file.write("DONE")

    print(f"DONE. Find at {full_path}.")


if __name__ == "__main__":
    main()