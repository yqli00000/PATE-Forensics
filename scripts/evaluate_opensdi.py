#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from torch.utils.data import DataLoader

import networks  # noqa: F401 - registers the existing DDL model
from utils.opensdi_metrics import OpenSDIMetricAccumulator
from data.opensdi_datamodule import OpenSDIRowGroupSampler
from data.opensdi_dataset import OpenSDIParquetDataset
from utils.common import load_config_with_cli, seed_everything
from utils.network_factory import get_model


MODEL_ORDER = ("sd15", "sd2", "sdxl", "sd3", "flux")
DISPLAY_NAMES = {"sd15": "SD1.5", "sd2": "SD2.1", "sdxl": "SDXL", "sd3": "SD3", "flux": "Flux.1"}


def _init_distributed(device_arg: str) -> tuple[int, int, torch.device]:
    world_size = int(os.getenv("WORLD_SIZE", "1") or 1)
    rank = int(os.getenv("RANK", "0") or 0)
    local_rank = int(os.getenv("LOCAL_RANK", "0") or 0)
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            requested = torch.device(device_arg)
            device = requested if requested.index is not None else torch.device("cuda:0")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, device


def _load_model(config_path: str, checkpoint_path: str, overrides: list[str], device: torch.device):
    conf = hydra.utils.instantiate(load_config_with_cli(config_path, args_list=overrides))
    model = get_model(conf)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not exactly match the configured DDL model: "
            f"missing={incompatible.missing_keys[:10]}, unexpected={incompatible.unexpected_keys[:10]}"
        )
    model.to(device).eval()
    return model


def _evaluate_generator(
    model: torch.nn.Module,
    *,
    data_root: str,
    generator: str,
    image_size: int,
    batch_size: int,
    workers: int,
    threshold: float,
    limit: int | None,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float | int]:
    dataset = OpenSDIParquetDataset(
        data_root,
        split="test",
        models=generator,
        image_size=image_size,
        train=False,
        augmentation="none",
        filter_mode="detection",
        limit=limit,
    )
    eval_dataset = dataset
    sampler = OpenSDIRowGroupSampler(dataset, shuffle=False, even_divisible=False)
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )
    metric = OpenSDIMetricAccumulator(threshold=threshold)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["pixel_values"].to(device, non_blocking=True)
            outputs = model(images)
            labels = batch["label"]
            scopes = batch["scope"]
            has_pixel_mask = batch["has_pixel_mask"].to(torch.bool)
            localization_selector = torch.tensor(
                [scope == "partial" for scope in scopes], dtype=torch.bool
            ) & (labels == 1) & has_pixel_mask
            metric.update(
                torch.sigmoid(outputs["logits"]),
                labels,
                outputs["pred_mask"],
                batch["mask"],
                localization_selector,
            )
            if rank == 0 and (batch_index + 1) % 50 == 0:
                print(f"[{DISPLAY_NAMES[generator]}] batches={batch_index + 1}/{len(loader)}", flush=True)
    metric.synchronize(device)
    result = metric.compute()
    return result


def _write_results(output_dir: Path, results: dict[str, dict[str, float | int]], metadata: dict) -> None:
    average_keys = ("detection_f1", "detection_accuracy", "localization_iou", "localization_f1")
    average = {
        key: sum(float(results[model][key]) for model in MODEL_ORDER) / len(MODEL_ORDER)
        for key in average_keys
    }
    payload = {"metadata": metadata, "per_generator": results, "average": average}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *average_keys, "detection_count", "localization_count"])
        for model in MODEL_ORDER:
            row = results[model]
            writer.writerow([DISPLAY_NAMES[model], *[row[key] for key in average_keys], row["detection_count"], row["localization_count"]])
        writer.writerow(["AVG", *[average[key] for key in average_keys], "", ""])

    loc_header = "| Method | " + " | ".join(f"{DISPLAY_NAMES[m]} IoU | {DISPLAY_NAMES[m]} F1" for m in MODEL_ORDER) + " | AVG IoU | AVG F1 |"
    loc_sep = "|---" * (2 * len(MODEL_ORDER) + 3) + "|"
    loc_values = []
    for model in MODEL_ORDER:
        loc_values.extend([f"{float(results[model]['localization_iou']):.4f}", f"{float(results[model]['localization_f1']):.4f}"])
    loc_row = "| DDL | " + " | ".join(loc_values + [f"{average['localization_iou']:.4f}", f"{average['localization_f1']:.4f}"]) + " |"

    det_header = "| Method | " + " | ".join(f"{DISPLAY_NAMES[m]} F1 | {DISPLAY_NAMES[m]} Acc" for m in MODEL_ORDER) + " | AVG F1 | AVG Acc |"
    det_sep = "|---" * (2 * len(MODEL_ORDER) + 3) + "|"
    det_values = []
    for model in MODEL_ORDER:
        det_values.extend([f"{float(results[model]['detection_f1']):.4f}", f"{float(results[model]['detection_accuracy']):.4f}"])
    det_row = "| DDL | " + " | ".join(det_values + [f"{average['detection_f1']:.4f}", f"{average['detection_accuracy']:.4f}"]) + " |"
    report = "\n".join(
        [
            "# OpenSDI evaluation", "", "## Localization (partial fake only)", "", loc_header, loc_sep, loc_row,
            "", "## Detection (all real/fake images)", "", det_header, det_sep, det_row, "",
            f"Fixed image/mask threshold: {metadata['threshold']:.2f}.",
        ]
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DDL checkpoint with the official OpenSDI protocol")
    parser.add_argument("--cfg", default=str(PROJECT_ROOT / "cfgs" / "train" / "train_opensdi.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="datasets/OpenSDI")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit-per-model", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args, overrides = parser.parse_known_args()
    if tuple(args.models) != MODEL_ORDER:
        raise ValueError("Formal reporting requires all five models in paper order; omit --models")

    rank, world_size, device = _init_distributed(args.device)
    seed_everything(42 + rank)
    model = _load_model(args.cfg, args.checkpoint, overrides, device)
    results = {}
    for generator in MODEL_ORDER:
        if rank == 0:
            print(f"Evaluating {DISPLAY_NAMES[generator]}...", flush=True)
        results[generator] = _evaluate_generator(
            model,
            data_root=args.data_root,
            generator=generator,
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            threshold=args.threshold,
            limit=args.limit_per_model,
            device=device,
            rank=rank,
            world_size=world_size,
        )

    if rank == 0:
        default_output = PROJECT_ROOT / "opensdi_test" / "results" / f"ddl_opensdi_{dt.datetime.now():%Y%m%d_%H_%M_%S}"
        output_dir = Path(args.output_dir) if args.output_dir else default_output
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_results(
            output_dir,
            results,
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "config": str(Path(args.cfg).resolve()),
                "data_root": str(Path(args.data_root).resolve()),
                "threshold": float(args.threshold),
                "image_size": int(args.image_size),
                "world_size": world_size,
                "evaluation_mode": "formal_full" if args.limit_per_model is None else "truncated_nonformal",
                "localization_filter": "label=1 AND scope=partial AND mask is non-null",
            },
        )
        print(json.dumps({"per_generator": results, "output_dir": str(output_dir)}, indent=2), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
