from __future__ import annotations
from typing import Any
import os
from logging import getLogger
import torch
from torch.optim import Adam
from torchvision.utils import make_grid
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric, MetricCollection
from .loss import NMI, Gradient
from loggers.img import ImageSave
if True:
    os.environ["NEURITE_BACKEND"] = "pytorch"
    os.environ["VXM_BACKEND"] = "pytorch"
    import neurite as ne
    from voxelmorph.torch.networks import VxmDense


logger = getLogger(__name__)


class VM(L.LightningModule):
    def __init__(self, inshape, lambda_R: float) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.inshape = inshape
        self.lambda_R = lambda_R
        self.net = VxmDense(inshape=inshape)
        self.L_sim = NMI(bins=32, xsigma=1.)
        self.L_smt = Gradient()
        metrics = MetricCollection({
            "loss": MeanMetric(),
            "l_sim": MeanMetric(),
            "l_smt": MeanMetric(),
        })
        self.metrics: dict[str, MetricCollection] = torch.nn.ModuleDict({
            "train_": metrics.clone("train/"),
            "val_": metrics.clone("val/"),
            "test_": metrics.clone("test/"),
        })  # type: ignore

    def forward(self, source, target, registration=False):
        return self.net(source, target, registration)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        b, *chwd = batch["M"].shape

        # Input Processing
        # Predict
        M_phi, flow = self(batch["M"], batch["F"])

        # Compute Loss
        l_sim = self.L_sim(M_phi, batch["F"])
        l_smt = self.L_smt(flow)
        loss = l_sim + self.lambda_R * l_smt

        # Update Metrics
        self.metrics["train_"]["l_sim"](l_sim.detach())
        self.metrics["train_"]["l_smt"](l_smt.detach())
        self.metrics["train_"]["loss"](loss.detach())

        if batch_idx == 0:
            self.log_image("train/M", batch["M"])
            self.log_image("train/F", batch["F"])
            self.log_image("train/M(φ)", M_phi)
            self.log_image("train/v", flow)

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT | None:
        b, *chwd = batch["M"].shape

        # Predict
        M_phi, flow = self(batch["M"], batch["F"])

        # Compute Loss
        l_sim = self.L_sim(M_phi, batch["F"])
        l_smt = self.L_smt(flow)
        loss = l_sim + self.lambda_R * l_smt

        # Update Metrics
        self.metrics["val_"]["l_sim"](l_sim.detach())
        self.metrics["val_"]["l_smt"](l_smt.detach())
        self.metrics["val_"]["loss"](loss.detach())

        if batch_idx == 0:
            self.log_image(f"val/{self.current_epoch}/M", batch["M"])
            self.log_image(f"val/{self.current_epoch}/F", batch["F"])
            self.log_image(f"val/{self.current_epoch}/M(φ)", M_phi)
            self.log_image(f"val/{self.current_epoch}/v", flow)

        return loss

    def test_step(self, batch: dict[str, Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT | None:
        """バッチ数 = 1"""
        M1 = batch["M1"]
        M2 = batch["M2"]
        M3 = batch["M3"]
        index1 = batch["MI1"]
        index2 = batch["MI2"]
        index3 = batch["MI3"]

        # 変形ベクトル場の計算
        M1_phi, phi1 = self(M1, M2, True)
        M2_phi, phi2 = self(M2, M3, True)
        M3_phi, phi3 = self(M3, M1, True)

        # 周回変形
        G = torch.meshgrid([torch.arange(x)
                           for x in self.inshape], indexing="ij")
        G = torch.stack(G, dim=0).unsqueeze(0)

        G_tilde = self.net.transformer(G, phi1)
        G_tilde = self.net.transformer(G_tilde, phi2)
        G_tilde = self.net.transformer(G_tilde, phi3)

        # 誤差
        loss = torch.square(G - G_tilde).mean()

        # ログ
        logger.info(f"{index1=}, {index2=}, {index3=} : {loss=}")

        self.log_image(f"test/{batch_idx}/M1", M1)
        self.log_image(f"test/{batch_idx}/M1(φ1)", M1_phi)
        self.log_image(f"test/{batch_idx}/M2", M2)
        self.log_image(f"test/{batch_idx}/M2(φ2)", M2_phi)
        self.log_image(f"test/{batch_idx}/M3", M3)
        self.log_image(f"test/{batch_idx}/M3(φ3)", M3_phi)
        self.log_image(f"test/{batch_idx}/φ1", phi1)
        self.log_image(f"test/{batch_idx}/φ2", phi2)
        self.log_image(f"test/{batch_idx}/φ3", phi3)

        return loss

    def on_train_epoch_end(self) -> None:
        # Log Metrics
        m = {k: v.compute() for k, v in self.metrics["train_"].items()}
        self.log_dict(m, prog_bar=True, sync_dist=True)
        # Reset
        self.metrics["train_"].reset()

    def on_validation_epoch_end(self) -> None:
        # Log Metrics
        m = {k: v.compute() for k, v in self.metrics["val_"].items()}
        self.log_dict(m, prog_bar=True, sync_dist=True)
        # Reset
        self.metrics["val_"].reset()

    def configure_optimizers(self):
        logger.info(f"{type(self.trainer.model)=}")
        param_size = map(
            lambda x: x.numel() if x.requires_grad else 0,
            self.trainer.model.parameters()
        )
        total_param_size = sum(param_size)
        logger.info(f"{total_param_size=}")
        return Adam(self.trainer.model.parameters(), 1e-5)

    def log_image(
        self,
        tag: str,
        img_tensor: Tensor,
    ):
        tb: SummaryWriter = self.loggers[0].experiment  # type: ignore
        _, _, d, *_ = img_tensor.shape
        img_tensor = img_tensor[:, :, d // 2]
        tile_img = make_grid(
            img_tensor,
            padding=1,
            normalize=True,
        ).clip(0., 1.)
        tb.add_image(tag, tile_img)

        if 2 < len(self.loggers) and isinstance(self.loggers[2], ImageSave):
            imgsave = self.loggers[2].experiment  # type: ignore
            imgsave.log_image(f"{tag}_%IDX%", img_tensor)

    def log_flow(
        self,
        tag: str,
        flow: Tensor,
    ):
        imsave: ImageSave = self.loggers[2].experiment
        _, _, d, *_ = flow.shape
        flow = flow[:, :, d // 2]
        flow = flow.permute(0, 2, 3, 1)
        ne.plot.flow(flow, title=tag)
