from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from utils.synthscars_explanations import (
    MULTICROP_COMPACT_PROMPT, data_url, parse_json_payload,
    render_explanation, unmodified_region_crops,
)


def load_localizations(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows.sort(key=lambda row: int(row["index"]))
    if len({str(row["uid"]) for row in rows}) != len(rows):
        raise ValueError("duplicate localization UIDs")
    rows = rows[:limit]
    if [int(row["index"]) for row in rows] != list(range(limit)):
        raise ValueError("localization must contain the first --limit samples in annotation order")
    for row in rows:
        for key in ("image_path", "pred_mask_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"{key} for UID {row['uid']}: {row[key]}")
    return rows


def build_content(focus: dict) -> tuple[list[dict], list[dict]]:
    image_path, mask_path = Path(focus["image_path"]), Path(focus["pred_mask_path"])
    crops, metadata = unmodified_region_crops(image_path, mask_path, 4, 8, 128)
    # The reported dataset images are PNG; preserve the historical API payload.
    content = [{"type": "image_url", "image_url": {"url": data_url(image_path.read_bytes(), "image/png")}}]
    content.extend({"type": "image_url", "image_url": {"url": data_url(crop, "image/png")}} for crop in crops)
    content.append({"type": "text", "text": MULTICROP_COMPACT_PROMPT})
    return content, metadata


def generate_one(focus: dict, *, key: str, api_url: str, api_model: str, retries: int) -> dict:
    content, metadata = build_content(focus)
    for attempt in range(retries):
        try:
            with OpenAI(api_key=key, base_url=api_url, timeout=120) as client:
                response = client.chat.completions.create(
                    model=api_model, messages=[{"role": "user", "content": content}],
                    temperature=0, max_tokens=500, response_format={"type": "json_object"},
                )
            raw = response.choices[0].message.content or ""
            explanation, structured = render_explanation(parse_json_payload(raw), max_artifacts=0)
            return {
                "uid": str(focus["uid"]), "image_name": focus["image_name"],
                "explanation": explanation, "structured_response": structured,
                "raw_response": raw, "api_model": api_model, "temperature": 0,
                "max_tokens": 500, "prompt_mode": "taxonomy_compact_multicrop",
                "region_crops": metadata,
            }
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("retries must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization-jsonl", type=Path, default=Path("outputs/synthscars_test/localization_per_sample.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/synthscars_test/explanations/predictions.jsonl"))
    parser.add_argument("--api-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-model", default="qwen3.5-flash")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and crop construction without API requests.")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        parser.error("workers and retries must be positive")
    rows = load_localizations(args.localization_jsonl, args.limit)
    if args.dry_run:
        crop_count = sum(len(build_content(row)[1]) for row in rows)
        print(json.dumps({"samples": len(rows), "crops": crop_count, "api_requests": 0}))
        return
    key = os.getenv(args.api_key_env)
    if not key:
        parser.error(f"missing API credential in {args.api_key_env}")
    manifest = {
        "prompt_mode": "taxonomy_compact_multicrop", "prompt": MULTICROP_COMPACT_PROMPT,
        "prompt_sha256": hashlib.sha256(MULTICROP_COMPACT_PROMPT.encode()).hexdigest(),
        "api_model": args.api_model, "api_url": args.api_url, "temperature": 0, "max_tokens": 500,
        "max_regions": 4, "min_component_area": 8, "min_crop_side": 128, "max_artifacts": 0,
        "inputs": "original + predicted-mask crops; no reference text, GT mask, overlay, or label",
        "localization_sha256": hashlib.sha256(args.localization_jsonl.read_bytes()).hexdigest(),
        "limit": args.limit,
    }
    manifest_path = args.output_jsonl.with_suffix(".manifest.json")
    if args.output_jsonl.exists() and not args.resume:
        parser.error("output exists; use --resume or choose another output path")
    done = set()
    if args.resume and args.output_jsonl.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            parser.error("resume requires a matching generation manifest")
        previous = [json.loads(line) for line in args.output_jsonl.read_text().splitlines() if line.strip()]
        done = {str(row["uid"]) for row in previous}
        if len(done) != len(previous) or not done.issubset({str(row["uid"]) for row in rows}):
            parser.error("duplicate or unexpected UIDs in existing output")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pending = [row for row in rows if str(row["uid"]) not in done]
    failures = 0
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_one, row, key=key, api_url=args.api_url,
                                   api_model=args.api_model, retries=args.retries): str(row["uid"]) for row in pending}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception as exc:
                    failures += 1
                    # Do not log credentials or full service responses.
                    print(f"Failed UID {futures[future]} ({type(exc).__name__}); retry with --resume")
                    continue
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
    if failures:
        raise SystemExit(f"{failures} samples failed; completed outputs were saved. Use --resume.")
    print(f"Completed {len(rows)} explanations: {args.output_jsonl}")


if __name__ == "__main__":
    main()
