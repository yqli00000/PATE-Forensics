#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import ListConfig

import engine
import networks  # noqa: F401 - registers the current DDL model
from data.opensdi_datamodule import OpenSDIDataModule
from utils.common import load_config_with_cli, seed_everything


def _device_configuration(conf) -> tuple[str, int, int, str]:
    accelerator = str(getattr(conf.train, "accelerator", "gpu"))
    gpu_ids = getattr(conf.train, "gpu_ids", [0])
    configured_devices = len(gpu_ids) if isinstance(gpu_ids, (list, tuple, ListConfig)) else int(gpu_ids)
    world_size = int(os.getenv("WORLD_SIZE", "1") or 1)
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1") or 1)
    launched = world_size > 1 and os.getenv("LOCAL_RANK") is not None
    devices = local_world_size if launched else configured_devices
    num_nodes = max(1, world_size // max(1, local_world_size)) if launched else 1
    strategy = str(getattr(conf.train, "strategy", "ddp" if devices > 1 else "auto"))
    if accelerator == "gpu" and not torch.cuda.is_available():
        accelerator, devices, num_nodes, strategy = "cpu", 1, 1, "auto"
    return accelerator, devices, num_nodes, strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the existing DDL model on local OpenSDI parquet shards")
    parser.add_argument("--cfg", default=str(PROJECT_ROOT / "cfgs" / "train" / "train_opensdi.yaml"))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--logdir", default=None, help="Run directory (default: opensdi_train/logs/<timestamp>)")
    args, overrides = parser.parse_known_args()

    conf = hydra.utils.instantiate(load_config_with_cli(args.cfg, args_list=overrides))
    seed_everything(int(conf.train.seed))
    torch.set_float32_matmul_precision("high")

    dm = OpenSDIDataModule(**dict(conf.opensdi_datamodule))
    pipeline = getattr(engine, str(conf.train.pipeline).split(".")[-1])
    lightning_model = pipeline(opt=conf)

    default_name = f"{conf.name}_{dt.datetime.now():%Y%m%d_%H_%M_%S}"
    run_dir = Path(args.logdir) if args.logdir else PROJECT_ROOT / "opensdi_train" / "logs" / default_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(run_dir), name="", version="")
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="{epoch:02d}-{val_mask_iou_epoch:.4f}",
        monitor=str(conf.train.monitor),
        mode=str(conf.train.monitor_mode),
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )

    accelerator, devices, num_nodes, strategy = _device_configuration(conf)
    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint],
        max_epochs=int(conf.train.train_epochs),
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=str(conf.train.precision),
        check_val_every_n_epoch=int(conf.train.check_val_every_n_epoch),
        log_every_n_steps=int(conf.train.log_every_n_steps),
        accumulate_grad_batches=int(conf.train.accumulation_steps),
        gradient_clip_val=float(conf.train.gradient_clip_val),
        gradient_clip_algorithm=str(conf.train.gradient_clip_algorithm),
        use_distributed_sampler=False,
    )
    trainer.fit(lightning_model, datamodule=dm, ckpt_path=args.resume)
    trainer.save_checkpoint(str(run_dir / "last.ckpt"))
    if trainer.is_global_zero:
        (run_dir / "best_checkpoint.txt").write_text(
            str(Path(checkpoint.best_model_path).resolve()) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
