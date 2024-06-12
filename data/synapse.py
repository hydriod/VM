from __future__ import annotations
import os
from pathlib import Path
from typing import Literal
import random
from logging import getLogger
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import SimpleITK as sitk
import numpy as np

SYNAPSE_SHAPE = (128, 128, 128)
CT_MIN = -809.0                 # 空気のCT値
logger = getLogger(__name__)


def _DataLoader(dataset: Dataset, shuffle: bool):
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=os.cpu_count() // 4,
        pin_memory=True,
        persistent_workers=True
    )


def _read_img(p: str):
    """.nii.gzを読み込んで正規化"""
    img = sitk.GetArrayFromImage(sitk.ReadImage(p))
    d, h, w = img.shape
    # リサイズ
    img = zoom(
        img,
        (SYNAPSE_SHAPE[0]/d, SYNAPSE_SHAPE[1]/h, SYNAPSE_SHAPE[2]/w)
    )
    # 正規化
    img = img.clip(CT_MIN, img.max())
    img = (img - img.min()) / (img.max() - img.min())
    # チャンネル軸追加
    img = img[np.newaxis]
    return img


class Synapse(L.LightningDataModule):
    def __init__(
        self,
        part: Literal["Cervix", "Abdomen"],
        root_dir: Path = Path("/Dataset/OpenSource/synapse"),
        test_times: int = 1
    ) -> None:
        super().__init__()
        self.root_dir = root_dir.joinpath(f"{part}/RawData")
        self.test_times = test_times

    def setup(self, stage: str) -> None:
        tr = self.root_dir.joinpath("Training/img")
        ts = self.root_dir.joinpath("Testing/img")

        tr = list(tr.glob("**/*.nii.gz"))
        ts = list(ts.glob("**/*.nii.gz"))

        self.train, self.val = train_test_split(tr, test_size=5, shuffle=False)
        self.test = ts

        # パスをログに出力
        if not (stage in ("test", "predict")):
            for i, p in enumerate(self.train):
                logger.info(f"train {i} : '{p}'")
            for i, p in enumerate(self.val):
                logger.info(f"val   {i} : '{p}'")
        else:
            for i, p in enumerate(self.test):
                logger.info(f"test  {i} : '{p}'")

    def train_dataloader(self):
        array = list(map(_read_img, self.train))
        array = np.stack(array, axis=0)
        return _DataLoader(dataset=TrainDS(array), shuffle=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        array = list(map(_read_img, self.val))
        array = np.stack(array, axis=0)
        return _DataLoader(dataset=TrainDS(array), shuffle=False)

    # def test_dataloader(self) -> TRAIN_DATALOADERS:
        # return _DataLoader(dataset=dataset, shuffle=False)


class TrainDS(Dataset):
    def __init__(self, array: list) -> None:
        super().__init__()
        self.array = torch.tensor(array, dtype=torch.float)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index) -> dict:
        index1 = index
        index2 = random.randint(0, len(self.array) - 1)

        img1 = self.array[index1]
        img2 = self.array[index2]

        # img1 = sitk.GetArrayFromImage(sitk.ReadImage(path1))
        # img2 = sitk.GetArrayFromImage(sitk.ReadImage(path2))

        return {"M": img1, "F": img2, "IM": index1, "IF": index2}


class TestDS(Dataset):
    def __init__(self, array: list, times: int = 1) -> None:
        super().__init__()
        self.array = array
        self.times = times

    def __len__(self):
        return len(self.array) * self.times

    def __getitem__(self, index) -> dict:
        index1 = index % len(self.array)
        index2 = random.randint(0, len(self.array) - 1)
        index3 = random.randint(0, len(self.array) - 1)

        img1 = self.array[index1]
        img2 = self.array[index2]
        img3 = self.array[index3]

        return {"M1": img1, "M2": img2, "M3": img3, "IM1": index1, "IM2": index2, "IM3": index3}
