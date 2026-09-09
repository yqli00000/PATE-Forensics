from __future__ import annotations

import argparse
import io
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torchvision import transforms

from infer import _load_model_from_checkpoint, choose_device
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image-forensics inference and export prediction JSON files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-images", type=int, default=None, help="Only process the first N sorted images.")
    parser.add_argument("--start-index", type=int, default=None, help="Start offset in the sorted image list, inclusive.")
    parser.add_argument("--end-index", type=int, default=None, help="End offset in the sorted image list, exclusive.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fake-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-box-area", type=int, default=16)
    parser.add_argument("--save-mask-png", action="store_true")
    parser.add_argument("--log-file", default=None, help="Write runtime logs here. Defaults to output-dir/infer_submission.log.")
    parser.add_argument("--summary-json", default=None, help="Write run summary here. Defaults to output-dir/infer_summary.json.")
    parser.add_argument("--score-jsonl", default=None, help="Write per-image logits/probabilities here. Defaults to output-dir/infer_scores.jsonl.")
    parser.add_argument("--reuse-existing-traces", action="store_true")
    parser.add_argument("--backbone-path", default="weights/dinov3-l16", help="DINOv3 backbone directory, relative to the repository root; overrides the path saved in checkpoint.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logger(output_dir: str | Path, log_file: str | None) -> logging.Logger:
    logger = logging.getLogger("infer_submission")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = Path(log_file) if log_file else Path(output_dir) / "infer_submission.log"
    ensure_dir(log_path.parent)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("log_file=%s", log_path)
    return logger


def collect_image_paths(image_path: str | None, image_dir: str | None) -> List[Path]:
    if image_path:
        return [Path(image_path)]
    if not image_dir:
        raise ValueError("One of --image-path or --image-dir must be provided.")

    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    image_paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise ValueError(f"No supported images found in {root}")
    return image_paths


def build_inference_transform(image_size: int) -> transforms.Compose:
    """
    Inference uses full-image resize to 512x512 by default.

    Note:
    - This avoids center crop, so no image region is discarded at test time.
    - It is slightly different from the fixed validation preprocessing used in training.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def classify_result(fake_probability: float, fake_threshold: float) -> str:
    """Main function 1: decide whether the image is real or fake."""
    return "fake" if float(fake_probability) >= float(fake_threshold) else "real"


def probability_to_logit(probability: float, eps: float = 1e-6) -> float:
    probability = min(1.0 - eps, max(eps, float(probability)))
    return float(math.log(probability / (1.0 - probability)))


def restore_mask_to_original_size(pred_mask_prob: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    """Resize the predicted mask back to the original image size."""
    return cv2.resize(
        np.squeeze(pred_mask_prob).astype(np.float32),
        (int(original_width), int(original_height)),
        interpolation=cv2.INTER_LINEAR,
    )


def normalize_box_to_submission(box: Sequence[int], original_width: int, original_height: int) -> List[int]:
    """
    Convert [x1, y1, x2, y2] from original-image coordinates into normalized 0–1000 coordinates.

    Coordinate convention:
    x' = round(x / W * 1000)
    y' = round(y / H * 1000)
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    return [
        int(round(x1 / max(1, original_width) * 1000)),
        int(round(y1 / max(1, original_height) * 1000)),
        int(round(x2 / max(1, original_width) * 1000)),
        int(round(y2 / max(1, original_height) * 1000)),
    ]


def compute_bounding_boxes(
    pred_mask_prob: np.ndarray,
    prediction: str,
    mask_threshold: float,
    min_box_area: int,
    original_width: int,
    original_height: int,
) -> List[List[int]]:
    """
    Main function 2:
    1. Restore the predicted mask to the original image size.
    2. Threshold the restored mask.
    3. Extract connected components.
    4. Convert each region into [x1, y1, x2, y2] and normalize to 0–1000 coordinates.
    """
    if prediction == "real":
        return []

    restored_mask_prob = restore_mask_to_original_size(pred_mask_prob, original_width, original_height)
    binary_mask = (restored_mask_prob >= float(mask_threshold)).astype(np.uint8)
    if binary_mask.sum() == 0:
        return []

    num_components, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    boxes: List[List[int]] = []
    for component_idx in range(1, num_components):
        x, y, w, h, area = stats[component_idx]
        if int(area) < int(min_box_area):
            continue
        boxes.append(
            normalize_box_to_submission(
                [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                original_width,
                original_height,
            )
        )
    return boxes


def _build_fallback_traces(prediction: str, fake_probability: float, boxes: Sequence[Sequence[int]]) -> str:
    if prediction == "real":
        return append_summary(
            "None. The image does not show obvious visible forgery traces. "
            "The lighting, edges, texture continuity, and overall physical consistency appear coherent.",
            prediction,
        )
    if not boxes:
        return append_summary(
            f"The image is predicted as fake with confidence {fake_probability:.4f}, "
            "but no stable localized visible forgery traces were extracted.",
            prediction,
        )
    return append_summary(
        f"The image is predicted as fake with confidence {fake_probability:.4f}. "
        f"Visible forgery traces are associated with {len(boxes)} localized suspicious region(s).",
        prediction,
    )


def strip_summary(text: str) -> str:
    return re.sub(r"\s*Summary:\s*This image has(?: not)? been tampered with\.?\s*$", "", text.strip(), flags=re.IGNORECASE)


def append_summary(text: str, prediction: str) -> str:
    text = strip_summary(text)
    if prediction == "fake":
        summary = "Summary: This image has been tampered with."
    else:
        summary = "Summary: This image has not been tampered with."
    return f"{text.rstrip()}\n\n{summary}"


def build_json_record(boxes: Sequence[Sequence[int]], traces: str, prediction: str) -> Dict[str, object]:
    return {
        "Bounding boxes": [list(box) for box in boxes],
        "Visible forgery traces": traces,
        "Classification result": prediction,
    }


def write_json_record(json_path: Path, boxes: Sequence[Sequence[int]], traces: str, prediction: str) -> None:
    record = build_json_record(boxes, traces, prediction)
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_traces(json_path: Path) -> str | None:
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    traces = payload.get("Visible forgery traces")
    if isinstance(traces, str) and traces.strip():
        return traces.strip()
    return None


def build_output_stem(name: str) -> str:
    """Use the source image filename stem for exported JSON and optional mask files."""
    safe = Path(name).stem.strip()
    return safe if safe else "sample"


def run_single_image_inference(
    model,
    device: torch.device,
    image_path: str | Path,
    *,
    output_dir: str | Path,
    image_size: int = 512,
    fake_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_box_area: int = 16,
    save_mask_png: bool = True,
    reuse_existing_traces: bool = False,
    logger: logging.Logger | None = None,
) -> Dict[str, object]:
    """
    Read one image, run inference, restore the predicted mask to original size,
    save the mask PNG, and return the prediction record plus file paths.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    json_dir = ensure_dir(output_dir / "json")
    mask_dir = ensure_dir(output_dir / "mask")

    image_transform = build_inference_transform(image_size)
    image_bytes = image_path.read_bytes()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_width, original_height = image.size

    with torch.no_grad():
        images = image_transform(image).unsqueeze(0).to(device)
        outputs = model(images)

        fake_logit = float(outputs["logits"].detach().cpu().numpy().reshape(-1)[0])
        fake_probability = float(torch.sigmoid(outputs["logits"]).detach().cpu().numpy().reshape(-1)[0])
        pred_mask_prob = outputs["pred_mask"].detach().cpu().numpy()[0]
        prediction = classify_result(fake_probability, fake_threshold)
        boxes = compute_bounding_boxes(
            pred_mask_prob,
            prediction,
            mask_threshold,
            min_box_area,
            original_width,
            original_height,
        )
    output_stem = build_output_stem(image_path.name)
    json_path = json_dir / f"{output_stem}.json"
    restored_mask_prob = restore_mask_to_original_size(pred_mask_prob, original_width, original_height)
    binary_mask = (restored_mask_prob >= mask_threshold).astype(np.uint8) * 255
    if prediction == "real":
        binary_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    mask_path = mask_dir / f"{output_stem}.png"
    mask_image = Image.fromarray(binary_mask)
    buffer = io.BytesIO()
    mask_image.save(buffer, format="PNG")
    mask_bytes = buffer.getvalue()
    if save_mask_png:
        mask_path.write_bytes(mask_bytes)

    traces = load_existing_traces(json_path) if reuse_existing_traces else None
    if traces is None:
        traces = _build_fallback_traces(prediction, fake_probability, boxes)
    write_json_record(json_path, boxes, traces, prediction)

    return {
        "image_path": str(image_path),
        "json_path": str(json_path),
        "mask_path": str(mask_path),
        "prediction": prediction,
        "fake_confidence": fake_probability,
        "fake_logit": fake_logit,
        "bounding_boxes": boxes,
        "visible_forgery_traces": traces,
    }


def run_batch_image_inference(
    model,
    device: torch.device,
    image_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    image_size: int = 512,
    fake_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_box_area: int = 16,
    save_mask_png: bool = True,
    reuse_existing_traces: bool = False,
    logger: logging.Logger | None = None,
) -> List[Dict[str, object]]:
    output_dir = Path(output_dir)
    json_dir = ensure_dir(output_dir / "json")
    mask_dir = ensure_dir(output_dir / "mask")
    image_transform = build_inference_transform(image_size)

    samples: List[Dict[str, Any]] = []
    tensors = []
    bad_results: List[Dict[str, object]] = []
    for image_path_like in image_paths:
        image_path = Path(image_path_like)
        try:
            image_bytes = image_path.read_bytes()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            message = f"[bad_image] path={image_path} error={type(exc).__name__}: {exc}"
            tqdm.write(message)
            if logger:
                logger.warning(message)
            output_stem = build_output_stem(image_path.name)
            json_path = json_dir / f"{output_stem}.json"
            mask_path = mask_dir / f"{output_stem}.png"
            traces = "Image could not be decoded; no visible forgery traces are available."
            write_json_record(json_path, [], traces, "real")
            bad_results.append(
                {
                    "image_path": str(image_path),
                    "json_path": str(json_path),
                    "mask_path": str(mask_path),
                    "prediction": "real",
                    "fake_confidence": 0.0,
                    "fake_logit": probability_to_logit(0.0),
                    "bounding_boxes": [],
                    "visible_forgery_traces": traces,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        original_width, original_height = image.size
        samples.append(
            {
                "image_path": image_path,
                "image_bytes": image_bytes,
                "original_width": original_width,
                "original_height": original_height,
            }
        )
        tensors.append(image_transform(image))

    if not tensors:
        return bad_results

    with torch.no_grad():
        images = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        outputs = model(images)
        fake_logits = outputs["logits"].detach().cpu().numpy().reshape(-1)
        fake_probabilities = torch.sigmoid(outputs["logits"]).detach().cpu().numpy().reshape(-1)
        pred_mask_probs = outputs["pred_mask"].detach().cpu().numpy()

    results: List[Dict[str, object]] = bad_results
    for sample_idx, sample in enumerate(samples):
        image_path = sample["image_path"]
        image_bytes = sample["image_bytes"]
        original_width = sample["original_width"]
        original_height = sample["original_height"]
        fake_logit = float(fake_logits[sample_idx])
        fake_probability = float(fake_probabilities[sample_idx])
        pred_mask_prob = pred_mask_probs[sample_idx]
        prediction = classify_result(fake_probability, fake_threshold)
        boxes = compute_bounding_boxes(
            pred_mask_prob,
            prediction,
            mask_threshold,
            min_box_area,
            original_width,
            original_height,
        )

        output_stem = build_output_stem(image_path.name)
        json_path = json_dir / f"{output_stem}.json"
        restored_mask_prob = restore_mask_to_original_size(pred_mask_prob, original_width, original_height)
        binary_mask = (restored_mask_prob >= mask_threshold).astype(np.uint8) * 255
        if prediction == "real":
            binary_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        mask_path = mask_dir / f"{output_stem}.png"
        mask_image = Image.fromarray(binary_mask)
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_bytes = buffer.getvalue()
        if save_mask_png:
            mask_path.write_bytes(mask_bytes)

        traces = load_existing_traces(json_path) if reuse_existing_traces else None
        if traces is None:
            traces = _build_fallback_traces(prediction, fake_probability, boxes)
        write_json_record(json_path, boxes, traces, prediction)
        results.append(
            {
                "image_path": str(image_path),
                "json_path": str(json_path),
                "mask_path": str(mask_path),
                "prediction": prediction,
                "fake_confidence": fake_probability,
                "fake_logit": fake_logit,
                "bounding_boxes": boxes,
                "visible_forgery_traces": traces,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    logger = configure_logger(args.output_dir, args.log_file)
    device = choose_device(args.device)
    model = _load_model_from_checkpoint(args.checkpoint, device,backbone_path=args.backbone_path,)
    model.eval()

    image_paths = collect_image_paths(args.image_path, args.image_dir)
    total_available_images = len(image_paths)
    if args.start_index is not None or args.end_index is not None:
        start_index = max(0, int(args.start_index or 0))
        end_index = None if args.end_index is None else max(start_index, int(args.end_index))
        image_paths = image_paths[start_index:end_index]
    elif args.limit_images is not None:
        image_paths = image_paths[: max(0, int(args.limit_images))]
    results = []
    batch_size = max(1, int(args.batch_size))
    with tqdm(total=len(image_paths), desc="Infer", dynamic_ncols=True) as progress:
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            results.extend(run_batch_image_inference(
                model,
                device,
                batch_paths,
                output_dir=args.output_dir,
                image_size=args.image_size,
                fake_threshold=args.fake_threshold,
                mask_threshold=args.mask_threshold,
                min_box_area=args.min_box_area,
                save_mask_png=args.save_mask_png,
                reuse_existing_traces=args.reuse_existing_traces,
                logger=logger,
            ))
            progress.update(len(batch_paths))

    summary_path = Path(args.summary_json) if args.summary_json else Path(args.output_dir) / "infer_summary.json"
    ensure_dir(summary_path.parent)
    num_fake = sum(1 for result in results if result["prediction"] == "fake")
    num_real = sum(1 for result in results if result["prediction"] == "real")
    fake_rate = num_fake / len(results) if results else 0.0

    summary = {
        "total_available_images": total_available_images,
        "limit_images": args.limit_images,
        "num_images": len(image_paths),
        "num_results": len(results),
        "num_fake": num_fake,
        "num_real": num_real,
        "sample_fake_rate": fake_rate,
        "fake_threshold": args.fake_threshold,
        "mask_threshold": args.mask_threshold,
        "min_box_area": args.min_box_area,
        "api_stats": {},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("summary_json=%s", summary_path)

    score_path = Path(args.score_jsonl) if args.score_jsonl else Path(args.output_dir) / "infer_scores.jsonl"
    ensure_dir(score_path.parent)
    with score_path.open("w", encoding="utf-8") as f:
        for result in sorted(results, key=lambda item: str(item.get("image_path", ""))):
            record = {
                "image_path": result.get("image_path"),
                "json_path": result.get("json_path"),
                "prediction": result.get("prediction"),
                "fake_probability": result.get("fake_confidence"),
                "fake_logit": result.get("fake_logit"),
                "num_boxes": len(result.get("bounding_boxes", []) or []),
                "fake_threshold": args.fake_threshold,
                "mask_threshold": args.mask_threshold,
                "min_box_area": args.min_box_area,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("score_jsonl=%s", score_path)


if __name__ == "__main__":
    main()
