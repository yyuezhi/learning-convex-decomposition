# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.GradScaler\(args\.\.\.\)` is deprecated\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"Starting from v1\.9\.0, `tensorboardX` has been removed as a dependency.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"predict returned None if it was on purpose, ignore this warning\.\.\.",
)

from model.config import default_argument_parser, setup
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint


class _LightningInfoFilter(logging.Filter):
    def filter(self, record):
        return "You are using a CUDA device" not in record.getMessage()


logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(_LightningInfoFilter())

import glob
import os, sys
import time
import torch

MESH_EXTENSIONS = (".obj", ".ply", ".stl", ".off", ".glb", ".gltf")


def _count_mesh_files(input_dir):
    count = 0
    for _, _, files in os.walk(input_dir):
        count += sum(filename.lower().endswith(MESH_EXTENSIONS) for filename in files)
    return count


def _resolve_inference_devices(cfg):
    if not torch.cuda.is_available():
        return 1
    gpu_count = torch.cuda.device_count()
    input_path = cfg.dataset.input_path
    if os.path.isfile(input_path):
        return 1
    if os.path.isdir(input_path):
        try:
            num_items = _count_mesh_files(input_path)
            if num_items > 0:
                return max(1, min(gpu_count, num_items))
        except Exception:
            pass
    return gpu_count

def train(cfg):
    seed_everything(cfg.seed)
    # snap shots
    
    checkpoint_callbacks = [ModelCheckpoint(
        monitor="train/current_epoch",
        dirpath=cfg.output_dir,
        filename="{epoch:02d}",
        save_top_k=100,
        save_last=True,
        every_n_epochs=cfg.save_every_epoch,
        mode="max",
        verbose=True
    )]

    ### we do not log inference
    predict_devices = _resolve_inference_devices(cfg)
    trainer = Trainer(devices=predict_devices,
                      accelerator="gpu",
                      precision="16-mixed",
                      strategy=DDPStrategy(find_unused_parameters=True) if predict_devices > 1 else "auto",
                      max_epochs=cfg.training_epochs,
                      log_every_n_steps=1,
                      limit_train_batches=3500,
                      limit_val_batches=None,
                      callbacks=checkpoint_callbacks,
                      inference_mode=(not cfg.recursive_split)
                     )


    from model.model_trainer import Model
    model = Model(cfg)        


    if cfg.ckpt_path is None:
        pattern = f"./ckpt/epoch=*.ckpt"
        ckpt_paths = sorted(glob.glob(pattern))

        found = False



        if not found:
            raise ValueError(f"No checkpoint found in {pattern}")


    base_dir = os.path.join( "exp_results",cfg.name)


    os.makedirs(base_dir, exist_ok=True)
    cfg.result_name = os.path.join(base_dir, cfg.result_name)
    os.makedirs(cfg.result_name, exist_ok=True)

    total_time = time.time()
    trainer.predict(model, ckpt_path=cfg.ckpt_path)

def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args, freeze=False)
    train(cfg)


if __name__ == '__main__':
    main()
