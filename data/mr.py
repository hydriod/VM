from __future__ import annotations
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS


def _DataLoader(dataset: Dataset, shuffle: bool):
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )


class MR_module(L.LightningDataModule):
    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self.root_dir = root_dir

    def setup(self, stage: str) -> None:
        return super().setup(stage)

    def train_dataloader(self):
        return _DataLoader(dataset=dataset, shuffle=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return _DataLoader(dataset=dataset, shuffle=False)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return _DataLoader(dataset=dataset, shuffle=False)
