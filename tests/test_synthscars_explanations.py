import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from scripts.generate_synthscars_explanations import build_content, generate_one, load_localizations
from utils.synthscars_explanations import (
    MULTICROP_COMPACT_PROMPT, parse_json_payload, render_explanation, unmodified_region_crops,
)
from utils.synthscars_protocol import rouge_l_f1


class SynthScarsExplanationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.mask = self.root / "mask.png"
        Image.new("RGB", (200, 200), "white").save(self.image)
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        mask[100:110, 100:110] = 255
        mask[180, 180] = 255
        Image.fromarray(mask).save(self.mask)
        self.focus = {"uid": "1", "index": 0, "image_name": "image.png",
                      "image_path": str(self.image), "pred_mask_path": str(self.mask),
                      "caption": "SECRET_REFERENCE", "predicted_label": "SECRET_LABEL"}

    def test_frozen_prompt(self):
        self.assertEqual(hashlib.sha256(MULTICROP_COMPACT_PROMPT.encode()).hexdigest(),
                         "3bb127286230aca109c1c7c813f2dc499ffee56438c07fd5cdd8e32005afb867")

    def test_crops_and_api_input(self):
        crops, metadata = unmodified_region_crops(self.image, self.mask, 4, 8, 128)
        self.assertEqual(len(crops), 2)
        self.assertEqual([item["area"] for item in metadata], [600, 100])
        self.assertEqual(metadata[0]["padded_box_xyxy"], [20, 12, 70, 48])
        self.assertEqual(min(metadata[0]["api_crop_size"]), 128)
        content, _ = build_content(self.focus)
        self.assertEqual([item["type"] for item in content], ["image_url"] * 3 + ["text"])
        self.assertNotIn("SECRET_REFERENCE", json.dumps(content))
        self.assertNotIn("SECRET_LABEL", json.dumps(content))

    def test_empty_mask(self):
        Image.new("L", (200, 200), 0).save(self.mask)
        content, metadata = build_content(self.focus)
        self.assertEqual(metadata, [])
        self.assertEqual(len(content), 2)

    def test_renderer(self):
        payload = parse_json_payload('```json\n{"scene":"A bird", "artifacts":[{"region":"wing.","abnormality":"is fused"}]}\n```')
        text, _ = render_explanation(payload, max_artifacts=0)
        self.assertEqual(text, "Upon examining the image. I have found: A bird. To elaborate, I have found the following artifacts. wing:is fused.")
        self.assertEqual(rouge_l_f1(text, text), 1)
        with self.assertRaises(ValueError):
            render_explanation({"scene": "A bird", "artifacts": []}, max_artifacts=0)

    def test_mock_api(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"scene":"A bird", "artifacts":[{"region":"wing","abnormality":"is fused"}]}'))])
        with patch("scripts.generate_synthscars_explanations.OpenAI", return_value=client):
            row = generate_one(self.focus, key="test", api_url="https://example.invalid", api_model="qwen3.5-flash", retries=1)
        self.assertEqual(row["uid"], "1")
        args = client.chat.completions.create.call_args.kwargs
        self.assertEqual(args["max_tokens"], 500)
        self.assertEqual(args["temperature"], 0)
        self.assertEqual(args["response_format"], {"type": "json_object"})

    def test_localization_validation(self):
        path = self.root / "localization.jsonl"
        path.write_text(json.dumps(self.focus) + "\n")
        self.assertEqual(len(load_localizations(path, 1)), 1)
        with self.assertRaises(ValueError):
            load_localizations(path, 2)
        path.write_text((json.dumps(self.focus) + "\n") * 2)
        with self.assertRaises(ValueError):
            load_localizations(path, 1)


if __name__ == "__main__":
    unittest.main()
