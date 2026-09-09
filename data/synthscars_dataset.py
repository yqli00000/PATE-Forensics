from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _stable_validation(uid: str, seed: int, val_ratio: float) -> bool:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{uid}".encode()).digest()[:8], "big")
    return value / 2**64 < val_ratio


def _load(annotation_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = []
    for index, wrapper in enumerate(payload):
        if not isinstance(wrapper, dict) or len(wrapper) != 1:
            raise ValueError(f"invalid singleton mapping at {annotation_path}:{index}")
        uid, record = next(iter(wrapper.items()))
        rows.append({"uid": str(uid), "image_name": record["img_file_name"], "refs": record["refs"]})
    return rows


def polygon_union(refs: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Match LEGION eval/utils.py::generate_mask exactly."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ref in refs:
        for polygon in ref.get("segmentation", []):
            points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    return mask


class SynthScarsTrainDataset(Dataset):
    """Use only official SynthScars/train, with a deterministic internal split."""

    def __init__(
        self,
        root: str | Path,
        *,
        train: bool,
        image_size: int = 512,
        val_ratio: float = 0.1,
        split_seed: int = 2026,
        limit: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "train/images"
        annotation = self.root / "train/annotations/train.json"
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be strictly between 0 and 1")
        all_rows = _load(annotation)
        # Never inspect or load test.json in this class.
        self.records = [row for row in all_rows if _stable_validation(row["uid"], split_seed, val_ratio) != train]
        self.records.sort(key=lambda row: row["uid"])
        if limit is not None:
            self.records = self.records[: int(limit)]
        self.image_size = int(image_size)
        missing = [r["image_name"] for r in self.records if not (self.image_dir / r["image_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing training images; first={missing[0]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.image_dir / record["image_name"]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        mask_array = polygon_union(record["refs"], height, width)
        mask = Image.fromarray(mask_array * 255, mode="L")
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR, antialias=True)
        image = TF.normalize(TF.to_tensor(image), MEAN, STD)
        mask = TF.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        mask = (TF.pil_to_tensor(mask) > 0).float()
        return {
            "uid": record["uid"], "image_name": record["image_name"], "image_path": str(path),
            "pixel_values": image, "mask": mask,
            # Positive-only image classification supervision, as in the recorded run.
            "label": torch.tensor(1, dtype=torch.long),
            "original_width": width, "original_height": height,
        }
