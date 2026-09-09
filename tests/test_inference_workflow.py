"""CPU regression checks with synthetic predictions and a mocked API client.

Run: python -m unittest discover -s tests -v
Optional: PATE_BASELINE_PATH=/path/to/old/infer_submission.py enables old/new comparison.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2  # Load native modules before temporarily patching sys.modules.
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Keep these tests independent of backbone weights and the API SDK.
    infer_stub = ModuleType("infer")
    infer_stub.choose_device = lambda name: torch.device("cpu")
    infer_stub._load_model_from_checkpoint = lambda *a, **k: None
    api_stub = ModuleType("openai")
    api_stub.OpenAI = lambda **kwargs: None
    with patch.dict(sys.modules, {"infer": infer_stub, "openai": api_stub}):
        spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
infer = load_script("local_infer", ROOT / "infer_submission.py")
refine = load_script("local_refine", ROOT / "update_json_traces.py")


class FixedModel(torch.nn.Module):
    def forward(self, images):
        # Black = real, white = fake; independent of batch composition.
        logits = images.mean(dim=(1, 2, 3)) * 3
        masks = torch.zeros((len(images), 1, 8, 8), device=images.device)
        masks[:, :, 2:6, 1:5] = 0.9
        return {"logits": logits, "pred_mask": masks}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = [self.root / "real.png", self.root / "fake.png"]
        for path, value in zip(self.paths, (0, 255)):
            Image.new("RGB", (40, 24), (value,) * 3).save(path)
        self.options = dict(image_size=8, min_box_area=1, save_mask_png=True)

    def test_single_batch_and_optional_baseline(self):
        modules = [infer]
        if os.getenv("PATE_BASELINE_PATH"):
            modules.append(load_script("baseline", os.environ["PATE_BASELINE_PATH"]))
        expected = None
        for index, module in enumerate(modules):
            for batch in (False, True):
                dest = self.root / f"out-{index}-{batch}"
                if batch:
                    results = module.run_batch_image_inference(
                        FixedModel(), torch.device("cpu"), self.paths,
                        output_dir=dest, **self.options)
                else:
                    results = [module.run_single_image_inference(
                        FixedModel(), torch.device("cpu"), path,
                        output_dir=dest, **self.options) for path in self.paths]
                records = [json.loads(Path(r["json_path"]).read_text()) for r in results]
                masks = [np.array(Image.open(r["mask_path"])) for r in results]
                scores = [(r["fake_logit"], r["fake_confidence"]) for r in results]
                self.assertEqual([r["Classification result"] for r in records], ["real", "fake"])
                self.assertEqual(records[0]["Bounding boxes"], [])
                self.assertTrue(records[1]["Bounding boxes"])
                self.assertEqual(masks[0].shape, (24, 40))
                self.assertFalse(masks[0].any())
                if expected is None:
                    expected = records, masks, scores
                else:
                    self.assertEqual(records, expected[0])
                    for actual, wanted in zip(masks, expected[1]):
                        np.testing.assert_array_equal(actual, wanted)
                    np.testing.assert_allclose(scores, expected[2], rtol=1e-6)
                saved = json.loads(Path(results[1]["json_path"]).read_text())
                saved["Visible forgery traces"] = "Existing explanation"
                Path(results[1]["json_path"]).write_text(json.dumps(saved))
                reused = module.run_single_image_inference(
                    FixedModel(), torch.device("cpu"), self.paths[1], output_dir=dest,
                    reuse_existing_traces=True, **self.options)
                self.assertEqual(reused["visible_forgery_traces"], "Existing explanation")

    def test_limit_images(self):
        dest = self.root / "limited"
        argv = ["infer_submission.py", "--checkpoint", "unused",
                "--image-dir", str(self.root), "--output-dir", str(dest),
                "--limit-images", "1", "--image-size", "8"]
        with patch.object(sys, "argv", argv), patch.object(
                infer, "_load_model_from_checkpoint", return_value=FixedModel()):
            infer.main()
        summary = json.loads((dest / "infer_summary.json").read_text())
        self.assertEqual(summary["num_images"], 1)
        self.assertEqual([p.name for p in (dest / "json").glob("*.json")], ["fake.json"])

    def call_api(self, stats, lock, limit=3):
        return refine.generate_visible_forgery_traces_from_old_text(
            b"image", "image.png", b"overlay", None, "Old description",
            "fake", True, api_url="https://example.invalid", api_key="test",
            api_model="mock", timeout=1, max_tokens=10, max_api_calls=limit,
            api_stats=stats, api_stats_lock=lock)

    def test_concurrent_request_limit_and_malformed_response(self):
        stats, lock = {}, threading.Lock()
        create = unittest.mock.Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="New description"))]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(refine, "OpenAI", return_value=client):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: self.call_api(stats, lock), range(32)))
        self.assertEqual(create.call_count, 3)
        self.assertEqual(stats["api_calls"], 3)
        create.return_value = SimpleNamespace(choices=[])
        stats = {}
        with patch.object(refine, "OpenAI", return_value=client):
            self.assertIn("Old description", self.call_api(stats, lock))
        self.assertEqual(stats["api_failed"], 1)

    def test_refinement_preserves_records_and_rejects_overwrite(self):
        dest = self.root / "outputs"
        infer.run_batch_image_inference(FixedModel(), torch.device("cpu"),
                                       self.paths, output_dir=dest, **self.options)
        options = dict(output_dir=dest, new_json_dir=dest / "refined",
                       api_url=None, api_key=None, api_model="mock", timeout=1,
                       max_tokens=10, max_api_calls=0, skip_empty_old_traces=False,
                       dry_run=False, api_stats={}, api_stats_lock=threading.Lock())
        for path in self.paths:
            source = dest / "json" / f"{path.stem}.json"
            before = source.read_bytes()
            refine.update_one_record(path, **options)
            self.assertEqual(source.read_bytes(), before)
            new = json.loads((dest / "refined" / source.name).read_text())
            old = json.loads(before)
            for key in ("Bounding boxes", "Classification result"):
                self.assertEqual(new[key], old[key])
        options["new_json_dir"] = dest / "json"
        with self.assertRaisesRegex(ValueError, "overwrite"):
            refine.update_one_record(self.paths[1], **options)


if __name__ == "__main__":
    unittest.main()
