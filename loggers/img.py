from __future__ import annotations
import os
from argparse import Namespace
from typing import Dict
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torchvision.utils import save_image
from lightning.pytorch.loggers import Logger


class ImageWriter:
    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        self.image_list: dict[str, Tensor] = {}
        self.flow_list: dict[str, Tensor] = {}
        return

    def log_image(self, tag: str, img: Tensor):
        img = img.detach().clone()
        b, c, d, *_ = img.shape
        img = img[:,:,d//2]
        # Normalize
        img = (img - img.min()) / (img.max() - img.min())
        # Concat time image
        if tag in self.image_list.keys():
            self.image_list[tag] = torch.cat([self.image_list[tag], img], dim=-1)
        else:
            self.image_list[tag] = img
        return
    
    def log_flow(self, tag: str, flow: Tensor):
        img = img.detach().clone()

    def save(self):
        for k, v in self.image_list.items():
            fp = os.path.join(self._log_dir, k)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            # Save image each batch
            for i in range(v.shape[0]):
                img = v[i]
                save_image(img, f"{fp}.png".replace("%IDX%", f"{i}"))
        self.image_list = {}
        for k, v in self.flow_list.items():
            fp = os.path.join(self._log_dir, k)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    img = v[i, j:j+1].permute(0, 2, 3, 1).cpu().numpy()
                    fig, axes = ne.plot.flow(img)
                    fig.savefig(f"{fp}.png".replace("%IDX%", f"{i}_{j}"))
                    fig.savefig(f"{fp}.svg".replace("%IDX%", f"{i}_{j}"))
                    plt.close(fig)
        self.flow_list = {}
        return


class ImageSave(Logger):
    def __init__(
        self,
        root_dir: str,
        name: str = "default",
        version: int | str | None = None,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._experiment = None

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int | str:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def log_dir(self) -> str:
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)

    def log_hyperparams(self, params: Namespace, *args, **kwargs):
        return super().log_hyperparams(params, *args, **kwargs)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None):
        return super().log_metrics(metrics, step)

    def save(self) -> None:
        if self._experiment is not None:
            self.experiment.save()

    @property
    def experiment(self) -> ImageWriter:
        """Return the experiment object associated with this logger."""
        if self._experiment is None:
            self._experiment = ImageWriter(self.log_dir)
        return self._experiment

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not os.path.isdir(versions_root):
            # log.warning("Missing logger folder: %s", versions_root)
            return 0

        existing_versions = []
        for name in os.listdir(versions_root):
            full_path = os.path.join(versions_root, name)
            if os.path.isdir(full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
