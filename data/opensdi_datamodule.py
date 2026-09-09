from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator

import lightning as L
import torch
from torch.utils.data import DataLoader, Sampler

from .opensdi_dataset import OpenSDIParquetDataset


def _distributed_context() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class OpenSDIRowGroupSampler(Sampler[int]):
    """Shuffle parquet row groups while keeping reads cache-friendly."""

    def __init__(
        self,
        dataset: OpenSDIParquetDataset,
        *,
        shuffle: bool,
        seed: int = 42,
        even_divisible: bool = True,
    ) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.even_divisible = bool(even_divisible)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _indices_for_rank(self) -> list[int]:
        rank, world_size = _distributed_context()
        rng = random.Random(self.seed + self.epoch)
        groups = list(self.dataset.row_group_spans)
        if self.shuffle:
            rng.shuffle(groups)
        ordered_indices: list[int] = []
        for start, end in groups:
            group_indices = list(range(start, end))
            if self.shuffle:
                rng.shuffle(group_indices)
            ordered_indices.extend(group_indices)

        if world_size == 1:
            return ordered_indices
        if self.even_divisible:
            samples_per_rank = math.ceil(len(ordered_indices) / world_size)
            total_size = samples_per_rank * world_size
            if total_size > len(ordered_indices):
                ordered_indices.extend(ordered_indices[: total_size - len(ordered_indices)])
            start = rank * samples_per_rank
            return ordered_indices[start : start + samples_per_rank]

        # Exact evaluation partition: no padding and therefore no duplicate
        # samples. A rank may receive one more item than another.
        start = len(ordered_indices) * rank // world_size
        end = len(ordered_indices) * (rank + 1) // world_size
        return ordered_indices[start:end]

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices_for_rank())

    def __len__(self) -> int:
        _, world_size = _distributed_context()
        if self.even_divisible:
            return math.ceil(len(self.dataset) / world_size)
        rank, _ = _distributed_context()
        start = len(self.dataset) * rank // world_size
        end = len(self.dataset) * (rank + 1) // world_size
        return end - start


class OpenSDIDataModule(L.LightningDataModule):
    """Lightning DataModule for official SD1.5 training and SD1.5 validation."""

    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int = 512,
        augmentation: str = "paper",
        batch_size: int = 2,
        val_batch_size: int = 4,
        num_workers: int = 4,
        val_num_workers: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        seed: int = 42,
        train_limit: int | None = None,
        val_limit: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.image_size = int(image_size)
        self.augmentation = str(augmentation)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.val_num_workers = int(val_num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.seed = int(seed)
        self.train_limit = train_limit
        self.val_limit = val_limit
        self.train_dataset: OpenSDIParquetDataset | None = None
        self.val_dataset: OpenSDIParquetDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"} and self.train_dataset is None:
            self.train_dataset = OpenSDIParquetDataset(
                self.root,
                split="train",
                models="sd15",
                image_size=self.image_size,
                train=True,
                augmentation=self.augmentation,
                filter_mode="all",
                limit=self.train_limit,
                seed=self.seed,
            )
            # This intentionally follows the released OpenSDI training code:
            # validation is the SD1.5 partial-fake test subset (5K samples).
            self.val_dataset = OpenSDIParquetDataset(
                self.root,
                split="test",
                models="sd15",
                image_size=self.image_size,
                train=False,
                augmentation="none",
                filter_mode="localization",
                limit=self.val_limit,
                seed=self.seed,
            )

    @staticmethod
    def _loader(
        dataset: OpenSDIParquetDataset,
        *,
        batch_size: int,
        workers: int,
        shuffle: bool,
        seed: int,
        pin_memory: bool,
        persistent_workers: bool,
    ) -> DataLoader:
        sampler = OpenSDIRowGroupSampler(dataset, shuffle=shuffle, seed=seed, even_divisible=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin_memory,
            persistent_workers=workers > 0 and persistent_workers,
            prefetch_factor=2 if workers > 0 else None,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return self._loader(
            self.train_dataset,
            batch_size=self.batch_size,
            workers=self.num_workers,
            shuffle=True,
            seed=self.seed,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        assert self.val_dataset is not None
        return self._loader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            workers=self.val_num_workers,
            shuffle=False,
            seed=self.seed,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
