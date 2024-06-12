from __future__ import annotations
import argparse
from pathlib import Path
from logging import getLogger
from torch.cuda.amp import GradScaler
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.plugins import FSDPPrecision, MixedPrecision
import models
import utils
import data


# Argument
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dir", type=Path, default="tmp")
parser.add_argument("-g", "--gpus", type=int, nargs="*", default=[3])
parser.add_argument("-l", "--lambda_R", type=float, default=1.)
parser.add_argument("-r", "--resume", type=Path, default=None)

args = parser.parse_args()
dir: Path = Path("experiments").joinpath(args.dir)
gpus: list[int] | int = args.gpus
lambda_R: float = args.lambda_R
resume: Path = args.resume

# logger
utils.setup_logger(filepath=dir.joinpath("root.log"))
logger = getLogger(__name__)

# Trainer
dir.joinpath("ckpt").mkdir(parents=True, exist_ok=True)
callbacks = [
    ModelCheckpoint(
        dirpath=dir.joinpath("ckpt"),
        filename="best_{epoch}",
        monitor="val/loss",
        every_n_epochs=1,
        save_top_k=3,
        save_weights_only=True,
    ),
    ModelCheckpoint(
        dirpath=dir.joinpath("ckpt"),
        filename="{epoch}",
        save_last=True,
        every_n_epochs=1,
        save_weights_only=True
    ),
    # EarlyStopping(
        # monitor="val/loss",
        # patience=5,
    # )
]
plugins = [
    # FSDPPrecision("16-mixed")
    MixedPrecision("16-mixed", "cuda")
]
trainer_loggers = [
    TensorBoardLogger(dir, "tb_loggers"),
    CSVLogger(dir, "csv_logger"),
]
trainer = L.Trainer(
    accelerator="gpu",
    strategy="ddp",
    devices=gpus,
    logger=trainer_loggers,
    callbacks=callbacks,
    max_epochs=-1,
    check_val_every_n_epoch=50,
    benchmark=True,
    plugins=plugins,
)

# データモジュール
datamodule = data.Synapse("Abdomen")

# モデルをロード
if resume is None:
    model = models.VM(data.SYNAPSE_SHAPE, lambda_R)
else:
    model = models.VM.load_from_checkpoint(str(resume))

trainer.fit(
    model=model,
    datamodule=datamodule,
)
