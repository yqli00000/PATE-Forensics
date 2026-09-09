from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from infer import _load_model_from_checkpoint, choose_device
from utils.synthscars_protocol import confusion, load_records, metrics_from_confusion, sum_confusions, union_polygon_mask

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen DDL-X in-domain localization on SynthScars test.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--backbone-path", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-threshold", type=float, default=0.5,
                    help="Mask decision threshold for the release checkpoint.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    image_size, fake_threshold, mask_threshold = 512, 0.5, args.mask_threshold
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask threshold must be in [0, 1]")
    output = args.output_dir
    masks = output / "pred_masks"
    probs = output / "mask_probabilities"
    masks.mkdir(parents=True, exist_ok=True); probs.mkdir(parents=True, exist_ok=True)
    records = load_records(args.data_root / "test/annotations/test.json")
    if args.limit <= 0 or args.limit > len(records): raise ValueError("limit must be in [1, 1000]")
    records = records[:args.limit]
    device = choose_device(args.device)
    if args.device == "cuda" and device.type != "cuda": raise RuntimeError("CUDA requested but unavailable")
    model = _load_model_from_checkpoint(str(args.checkpoint), device, backbone_path=str(args.backbone_path))
    model.eval()
    row_path = output / "localization_per_sample.jsonl"
    rows = []
    if args.resume and row_path.exists():
        rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {row["uid"] for row in rows}
    if not args.resume:
        row_path.write_text("", encoding="utf-8")
    with torch.inference_mode():
        for index, record in enumerate(records):
            if record.uid in completed:
                continue
            image_path = args.data_root / "test/images" / record.image_name
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            w, h = image.size
            tensor = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
            tensor = TF.normalize(TF.to_tensor(tensor), MEAN, STD).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(tensor)
            fake_probability = float(torch.sigmoid(result["logits"])[0].float().cpu())
            probability_512 = result["pred_mask"][0, 0].float().cpu().numpy()
            probability = cv2.resize(probability_512, (w, h), interpolation=cv2.INTER_LINEAR)
            prediction = probability >= mask_threshold
            gt = union_polygon_mask(record.refs, h, w).astype(bool)
            c = confusion(prediction, gt); m = metrics_from_confusion(c)
            mask_path = masks / f"{record.uid}.png"
            probability_path = probs / f"{record.uid}.npz"
            cv2.imwrite(str(mask_path), prediction.astype(np.uint8) * 255)
            np.savez_compressed(probability_path, probability=probability.astype(np.float32))
            row = {"index": index, "uid": record.uid, "image_name": record.image_name,
                         "image_path": str(image_path.resolve()), "height": h, "width": w,
                         "predicted_label": "fake" if fake_probability >= fake_threshold else "real",
                         "fake_probability": fake_probability, "predicted_pixels": int(prediction.sum()),
                         "gt_pixels": int(gt.sum()), **c, **m, "pred_mask_path": str(mask_path.resolve()),
                         "mask_probability_path": str(probability_path.resolve())}
            rows.append(row)
            with row_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    total = sum_confusions(rows); aggregate = metrics_from_confusion(total)
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    summary = {"protocol": "SynthScars official test; in-domain; union binary masks; aggregate pixel confusion",
               "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
               "git_commit": commit, "num_samples": len(rows), "selection": f"first {len(rows)} records in official test.json order",
               "image_size": image_size, "fake_threshold": fake_threshold, "mask_threshold": mask_threshold,
               "post_processing": f"bilinear restore probability to original W,H; >={mask_threshold}; no classification gating; no component removal",
               "coordinate_space": "original image pixels", "confusion": total, **aggregate,
               "scale_0_100": {k: 100*v for k,v in aggregate.items()}}
    if len(rows) != len(records):
        raise RuntimeError(f"incomplete run: {len(rows)} rows for {len(records)} records")
    (output / "localization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
