from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by the command-line error path
    raise ImportError(
        "OpenSDI parquet loading requires pyarrow. Install requirements.txt "
        "in the DDL environment."
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_MODELS = ("sd15", "sd2", "sdxl", "sd3", "flux")


@dataclass(frozen=True)
class OpenSDIRecord:
    file_index: int
    row_group: int
    row_in_group: int
    key: str
    label: int
    scope: str


def _scope_from_key(key: str) -> str:
    prefix = key.split("/", 1)[0].lower() if "/" in key else ""
    if prefix == "partial":
        return "partial"
    if prefix == "entire":
        return "entire"
    # The SD1.5 test split has 2K real + 2K fake legacy rows whose keys are
    # bare PNG names. Their masks are null, so they belong to image detection,
    # not partial-forgery localization.
    return "legacy_entire"


def _decode_hf_image(value: Any, parquet_path: Path) -> Image.Image | None:
    """Decode Hugging Face Image structs (``{bytes, path}``) without datasets."""
    if value is None:
        return None
    raw = value.get("bytes") if isinstance(value, dict) else None
    relative_path = value.get("path") if isinstance(value, dict) else None
    if raw is not None:
        with Image.open(io.BytesIO(raw)) as opened:
            return opened.copy()
    if relative_path:
        candidate = parquet_path.parent / str(relative_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"Parquet image path does not exist: {candidate}")
        with Image.open(candidate) as opened:
            return opened.copy()
    raise ValueError(f"Invalid Hugging Face Image value in {parquet_path}: {value!r}")


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as opened:
        return opened.convert("RGB").copy()


class OpenSDIParquetDataset(Dataset):
    """Lazy, row-group-cached reader for the local OpenSDI parquet shards.

    The official Hugging Face rows contain ``key``, ``image``, ``mask`` and
    ``label``. Null masks are interpreted exactly as the official loader:
    real -> all-zero mask; fake -> all-one mask. Localization evaluation uses
    ``filter_mode='localization'``, which retains only partial fake rows with
    genuine pixel annotations.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        models: str | Iterable[str] | None = None,
        image_size: int = 512,
        train: bool = False,
        augmentation: str = "paper",
        filter_mode: str = "all",
        limit: int | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.split = str(split).lower()
        if self.split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.data_dir = self.root / ("OpenSDI_train" if self.split == "train" else "OpenSDI_test") / "data"
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"OpenSDI parquet directory not found: {self.data_dir}")

        if models is None:
            selected_models = ("sd15",) if self.split == "train" else SUPPORTED_MODELS
        elif isinstance(models, str):
            selected_models = (models,)
        else:
            selected_models = tuple(str(item) for item in models)
        unknown = sorted(set(selected_models) - set(SUPPORTED_MODELS))
        if unknown:
            raise ValueError(f"Unsupported OpenSDI model split(s): {unknown}")
        if self.split == "train" and set(selected_models) != {"sd15"}:
            raise ValueError("The official OpenSDI training set contains SD1.5 only")

        self.files: list[Path] = []
        for model in selected_models:
            self.files.extend(sorted(self.data_dir.glob(f"{model}-*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"No matching parquet shards under {self.data_dir}")

        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.train = bool(train)
        self.augmentation = str(augmentation).lower()
        if self.augmentation not in {"paper", "github", "none"}:
            raise ValueError("augmentation must be one of: paper, github, none")
        self.filter_mode = str(filter_mode).lower()
        valid_filters = {"all", "detection", "localization", "partial_fake", "entire_fake", "real"}
        if self.filter_mode not in valid_filters:
            raise ValueError(f"filter_mode must be one of {sorted(valid_filters)}")
        self.seed = int(seed)

        self.records: list[OpenSDIRecord] = []
        self.row_group_spans: list[tuple[int, int]] = []
        self._build_index(limit=None if limit is None else max(0, int(limit)))
        if not self.records:
            raise ValueError("OpenSDI dataset is empty after filtering/limit")

        self._parquet_handles: dict[int, Any] = {}
        self._cached_group_key: tuple[int, int] | None = None
        self._cached_group_rows: list[dict[str, Any]] | None = None

    def _accept(self, key: str, label: int) -> bool:
        scope = _scope_from_key(key)
        if self.filter_mode in {"all", "detection"}:
            return True
        if self.filter_mode in {"localization", "partial_fake"}:
            return label == 1 and scope == "partial"
        if self.filter_mode == "entire_fake":
            return label == 1 and scope in {"entire", "legacy_entire"}
        if self.filter_mode == "real":
            return label == 0
        return False

    def _build_index(self, limit: int | None) -> None:
        for file_index, path in enumerate(self.files):
            parquet_file = pq.ParquetFile(path)
            for group_index in range(parquet_file.metadata.num_row_groups):
                metadata_rows = parquet_file.read_row_group(group_index, columns=["key", "label"]).to_pylist()
                group_start = len(self.records)
                for row_in_group, row in enumerate(metadata_rows):
                    key = str(row["key"])
                    label = int(row["label"])
                    if not self._accept(key, label):
                        continue
                    self.records.append(
                        OpenSDIRecord(
                            file_index=file_index,
                            row_group=group_index,
                            row_in_group=row_in_group,
                            key=key,
                            label=label,
                            scope=_scope_from_key(key),
                        )
                    )
                    if limit is not None and len(self.records) >= limit:
                        break
                group_end = len(self.records)
                if group_end > group_start:
                    self.row_group_spans.append((group_start, group_end))
                if limit is not None and len(self.records) >= limit:
                    return

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_parquet_handles"] = {}
        state["_cached_group_key"] = None
        state["_cached_group_rows"] = None
        return state

    def _read_raw_row(self, record: OpenSDIRecord) -> dict[str, Any]:
        cache_key = (record.file_index, record.row_group)
        if self._cached_group_key != cache_key or self._cached_group_rows is None:
            handle = self._parquet_handles.get(record.file_index)
            if handle is None:
                handle = pq.ParquetFile(self.files[record.file_index])
                self._parquet_handles[record.file_index] = handle
            table = handle.read_row_group(
                record.row_group,
                columns=["key", "image", "mask", "label"],
            )
            self._cached_group_rows = table.to_pylist()
            self._cached_group_key = cache_key
        return self._cached_group_rows[record.row_in_group]

    @staticmethod
    def _resize_pair(image: Image.Image, mask: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
        height, width = size
        image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [height, width], interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _paper_augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Augmentations listed in the CVPR paper, including its random crop."""
        scale = random.uniform(0.8, 1.2)
        scaled_h = max(1, round(image.height * scale))
        scaled_w = max(1, round(image.width * scale))
        image, mask = self._resize_pair(image, mask, (scaled_h, scaled_w))

        pad_w = max(0, self.image_size - image.width)
        pad_h = max(0, self.image_size - image.height)
        if pad_w or pad_h:
            left = random.randint(0, pad_w) if pad_w else 0
            top = random.randint(0, pad_h) if pad_h else 0
            padding = (left, top, pad_w - left, pad_h - top)
            image = ImageOps.expand(image, border=padding, fill=0)
            mask = ImageOps.expand(mask, border=padding, fill=0)
        top = random.randint(0, max(0, image.height - self.image_size))
        left = random.randint(0, max(0, image.width - self.image_size))
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        if random.random() < 0.2:
            image = _jpeg_roundtrip(image, random.randint(70, 100))
        return image, mask

    def _github_augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Released train.py augmentation (which differs from the paper)."""
        scale = random.uniform(0.8, 1.2)
        image, mask = self._resize_pair(
            image,
            mask,
            (max(1, round(image.height * scale)), max(1, round(image.width * scale))),
        )
        if random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.2:
            image = _jpeg_roundtrip(image, random.randint(70, 100))
        if random.random() < 0.5:
            turns = random.randint(1, 3)
            image = image.rotate(90 * turns, expand=True)
            mask = mask.rotate(90 * turns, expand=True)
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
        return self._resize_pair(image, mask, (self.image_size, self.image_size))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        raw_row = self._read_raw_row(record)
        parquet_path = self.files[record.file_index]
        image = _decode_hf_image(raw_row["image"], parquet_path)
        if image is None:
            raise ValueError(f"Null image at key={record.key!r} in {parquet_path}")
        image = image.convert("RGB")
        original_width, original_height = image.size

        decoded_mask = _decode_hf_image(raw_row["mask"], parquet_path)
        has_pixel_mask = decoded_mask is not None
        if decoded_mask is None:
            mask = Image.new("L", image.size, color=0 if record.label == 0 else 255)
        else:
            mask = decoded_mask.convert("L")
            if mask.size != image.size:
                raise ValueError(
                    f"Image/mask size mismatch for {record.key}: image={image.size}, mask={mask.size}"
                )

        label = record.label
        if self.train and self.augmentation == "paper":
            image, mask = self._paper_augment(image, mask)
        elif self.train and self.augmentation == "github":
            image, mask = self._github_augment(image, mask)
        else:
            image, mask = self._resize_pair(image, mask, (self.image_size, self.image_size))

        mask_tensor = (TF.pil_to_tensor(mask) > 127).to(torch.float32)
        # The official loader recomputes the post-augmentation label from the
        # transformed mask, so a crop that removes a partial forgery is real.
        if self.train:
            label = int(mask_tensor.any().item())
        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

        return {
            "record_index": int(index),
            "uid": record.key,
            "key": record.key,
            "generator": self.files[record.file_index].name.split("-", 1)[0],
            "scope": record.scope,
            "has_pixel_mask": bool(has_pixel_mask),
            "pixel_values": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "original_width": original_width,
            "original_height": original_height,
        }
