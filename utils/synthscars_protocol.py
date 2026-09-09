from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class Record:
    uid: str
    image_name: str
    caption: str
    refs: tuple[dict[str, Any], ...]


def load_records(annotation_path: Path) -> list[Record]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("test.json must be a JSON list")
    out: list[Record] = []
    seen: set[str] = set()
    for index, wrapper in enumerate(payload):
        if not isinstance(wrapper, dict) or len(wrapper) != 1:
            raise ValueError(f"item {index} is not a singleton id-to-record mapping")
        uid, value = next(iter(wrapper.items()))
        if str(uid) in seen or not isinstance(value, dict):
            raise ValueError(f"invalid/duplicate record at item {index}")
        seen.add(str(uid))
        refs = value.get("refs")
        if not isinstance(refs, list):
            raise ValueError(f"record {uid} has invalid refs")
        out.append(Record(str(uid), str(value["img_file_name"]), str(value.get("caption", "")), tuple(refs)))
    return out


def union_polygon_mask(refs: Sequence[dict[str, Any]], height: int, width: int) -> np.ndarray:
    """Match LEGION eval/utils.py: float polygon -> int32 -> cv2.fillPoly, union=1."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ref in refs:
        for polygon in ref.get("segmentation", []):
            points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    return mask


def confusion(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    p, g = pred.astype(bool), gt.astype(bool)
    return {
        "tp": int(np.logical_and(p, g).sum()),
        "fp": int(np.logical_and(p, ~g).sum()),
        "fn": int(np.logical_and(~p, g).sum()),
        "tn": int(np.logical_and(~p, ~g).sum()),
    }


def metrics_from_confusion(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    fg_iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    bg_iou = tn / (tn + fp + fn) if tn + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"foreground_iou": fg_iou, "background_iou": bg_iou, "miou": (fg_iou + bg_iou) / 2, "f1": f1}


def sum_confusions(rows: Iterable[dict[str, int]]) -> dict[str, int]:
    return {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn", "tn")}


def normalize_text(text: str) -> str:
    """Identical deterministic preprocessing for reference and prediction."""
    return re.sub(r"\s+", " ", text).strip()


def reference_explanation(record: Record) -> str:
    return normalize_text(record.caption)


def rouge_l_f1(prediction: str, reference: str) -> float:
    """Token-level ROUGE-L F1 (LCS), with shared whitespace normalization."""
    a, b = normalize_text(prediction).split(), normalize_text(reference).split()
    if not a or not b:
        return float(a == b)
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
