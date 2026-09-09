import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from data.opensdi_dataset import OpenSDIParquetDataset
from data.opensdi_datamodule import OpenSDIRowGroupSampler
from data.synthscars_dataset import SynthScarsTrainDataset, polygon_union
from utils.synthscars_protocol import union_polygon_mask, confusion, metrics_from_confusion


class AdditionalDatasetTests(unittest.TestCase):
    def test_synthscars_train_only_split_and_polygon_union(self):
        refs = [{"segmentation": [[1, 1, 4, 1, 4, 4, 1, 4], [3, 3, 6, 3, 6, 6, 3, 6]]}]
        expected = polygon_union(refs, 8, 8)
        self.assertEqual(int(expected.sum()), 28)
        np.testing.assert_array_equal(expected, union_polygon_mask(refs, 8, 8))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train/images").mkdir(parents=True)
            (root / "train/annotations").mkdir()
            Image.new("RGB", (8, 8)).save(root / "train/images/image.png")
            rows = [{str(i): {"img_file_name": "image.png", "refs": refs}} for i in range(100)]
            (root / "train/annotations/train.json").write_text(json.dumps(rows))
            train = SynthScarsTrainDataset(root, train=True, image_size=8)
            val = SynthScarsTrainDataset(root, train=False, image_size=8)
            train_ids = {r["uid"] for r in train.records}
            val_ids = {r["uid"] for r in val.records}
            self.assertFalse(train_ids & val_ids)
            self.assertEqual(len(train_ids | val_ids), 100)
            self.assertGreater(len(val), 0)
            sample = train[0]
            self.assertEqual(tuple(sample["pixel_values"].shape), (3, 8, 8))
            self.assertEqual(sample["label"].item(), 1)
            np.testing.assert_array_equal(sample["mask"][0].numpy(), expected)

    def test_opensdi_masks_filter_and_exact_distributed_partition(self):
        def encoded(image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return {"bytes": buffer.getvalue(), "path": None}

        rgb = encoded(Image.new("RGB", (8, 8)))
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:4] = 255
        rows = [
            {"key": "entire/real.png", "label": 0, "image": rgb, "mask": None},
            {"key": "entire/fake.png", "label": 1, "image": rgb, "mask": None},
            {"key": "partial/fake.png", "label": 1, "image": rgb, "mask": encoded(Image.fromarray(mask))},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "OpenSDI_test/data"
            data.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist(rows), data / "sd15-000.parquet", row_group_size=2)
            dataset = OpenSDIParquetDataset(tmp, split="test", models="sd15", image_size=8)
            self.assertEqual([int(dataset[i]["mask"].sum()) for i in range(3)], [0, 64, 32])
            localized = OpenSDIParquetDataset(tmp, split="test", models="sd15", image_size=8, filter_mode="localization")
            self.assertEqual(len(localized), 1)
            self.assertTrue(localized[0]["has_pixel_mask"])
            indices = []
            for rank in range(4):
                with patch("data.opensdi_datamodule._distributed_context", return_value=(rank, 4)):
                    indices.extend(OpenSDIRowGroupSampler(dataset, shuffle=False, even_divisible=False))
            self.assertEqual(sorted(indices), [0, 1, 2])

    def test_two_class_metrics(self):
        c = confusion(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        self.assertEqual(c, dict(tp=1, fp=1, fn=1, tn=1))
        metrics = metrics_from_confusion(c)
        self.assertAlmostEqual(metrics["miou"], 1 / 3)
        self.assertAlmostEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
