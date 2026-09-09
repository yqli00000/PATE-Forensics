from __future__ import annotations

import base64
import io
import re
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.synthscars_protocol import normalize_text


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


FIXED_PREFIX = "Upon examining the image. I have found:"


FIXED_BRIDGE = "To elaborate, I have found the following artifacts."


MULTICROP_COMPACT_PROMPT = """Write a localized synthetic-artifact description for the SynthScars benchmark.

The first supplied image is the complete original. The remaining supplied images, if any, are distinct unmodified crops from the same original, ordered from larger to smaller focus regions. They only indicate where to inspect and are not proof of artifacts. Inspect every crop and relate it back to the complete scene. Never mention images, crops, masks, overlays, boxes, ordering, models, predictions, prompts, confidence, or the inspection process.

Check the people, animals, objects, text, and scene elements actually present for six kinds of concrete local defects: (1) missing or incomplete parts; (2) extra or duplicated parts; (3) fused, intersecting, or incorrectly connected parts; (4) deformed, twisted, split, asymmetric, or structurally inconsistent parts; (5) distorted, incomplete, misspelled, or unreadable text and symbols; and (6) impossible local spatial relationships or clear localized material, texture, color, shadow, or reflection corruption.

Name each affected part precisely and state exactly what is visibly wrong. Do not report generic smoothness, general blur, lighting style, aesthetic quality, or overall synthetic appearance. Do not infer a defect merely from a focus crop.

Cover every distinct strongly supported artifact visible across the supplied regions. Merge two descriptions when they concern the same defect on the same part. Keep separate defects on paired parts, such as the left and right eyes or two different hands, as separate entries. Order entries by the prominence of the affected subject, then from larger to smaller focused region. Use short noun phrases for region names and concise defect statements without repeated scene wording. The scene sentence must mention the verified defects in the same order as the artifact entries. Usually one to four entries are sufficient.

Return JSON only. This is a format template, not a content example:
{"scene":"<concise factual scene sentence mentioning the verified defects>","artifacts":[{"region":"<specific affected object or part>","abnormality":"<specific visible defect>"}]}

Replace every angle-bracket field with current-image content. Do not output angle brackets, markdown, task explanations, or benchmark boilerplate."""


def unmodified_region_crops(image_path: Path, mask_path: Path, max_regions: int, min_area: int,
                            min_crop_side: int) -> tuple[list[bytes], list[dict]]:
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for label_id in range(1, count):
        x, y, width, height, area = map(int, stats[label_id])
        if area >= min_area:
            components.append((area, x, y, width, height))
    components.sort(reverse=True)
    crops, metadata = [], []
    for area, x, y, width, height in components[:max_regions]:
        pad_x, pad_y = max(8, width // 3), max(8, height // 3)
        left, top = max(0, x - pad_x), max(0, y - pad_y)
        right, bottom = min(image.width, x + width + pad_x), min(image.height, y + height + pad_y)
        crop = image.crop((left, top, right, bottom))
        if min(crop.size) <= 10:
            continue
        source_size = crop.size
        if min_crop_side > 0 and min(crop.size) < min_crop_side:
            scale = min_crop_side / min(crop.size)
            crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        crops.append(buffer.getvalue())
        metadata.append({"area": area, "bbox_xywh": [x, y, width, height],
                         "padded_box_xyxy": [left, top, right, bottom],
                         "source_crop_size": list(source_size), "api_crop_size": list(crop.size)})
    return crops, metadata


def clean_value(value: object) -> str:
    text = normalize_text(str(value or "")).strip(" `\"'")
    text = re.sub(r"^(?:Upon examining the image\. I have found:|To elaborate, I have found the following artifacts\.)\s*", "", text, flags=re.I)
    return text.strip()


def parse_json_payload(text: str) -> dict:
    candidate = text.strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
    candidate = re.sub(r"\s*```$", "", candidate)
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response contains no complete JSON object")
    value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON response is not an object")
    return value


def render_explanation(payload: dict, max_artifacts: int = 2) -> tuple[str, dict]:
    scene = clean_value(payload.get("scene"))
    raw_artifacts = payload.get("artifacts")
    if not scene:
        raise ValueError("empty scene in JSON response")
    if not isinstance(raw_artifacts, list):
        raise ValueError("artifacts is not a list")

    artifacts: list[dict[str, str]] = []
    selected = raw_artifacts if max_artifacts <= 0 else raw_artifacts[:max_artifacts]
    for raw in selected:
        if not isinstance(raw, dict):
            continue
        region = clean_value(raw.get("region"))
        abnormality = clean_value(raw.get("abnormality"))
        if region and abnormality:
            artifacts.append({"region": region.rstrip(".:"), "abnormality": abnormality.rstrip()})
    if not artifacts:
        raise ValueError("no valid artifact entries in JSON response")

    if scene[-1] not in ".!?":
        scene += "."
    entries = "".join(
        f" {item['region']}:{item['abnormality']}" + ("" if item["abnormality"][-1] in ".!?" else ".")
        for item in artifacts
    )
    explanation = normalize_text(f"{FIXED_PREFIX} {scene} {FIXED_BRIDGE}{entries}")
    return explanation, {"scene": scene, "artifacts": artifacts}
