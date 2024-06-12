from __future__ import annotations
import argparse
from pathlib import Path
from logging import getLogger
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.plugins import FSDPPrecision, MixedPrecision
from loggers import ImageSave
import models
import utils
import data


# Argument
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dir", type=Path, default="tmp")
parser.add_argument("-g", "--gpus", type=int, nargs="*", default=[2, 3])
parser.add_argument("-l", "--lambda_R", type=float, default=1.)

args = parser.parse_args()
dir: Path = Path("experiments").joinpath(args.dir)
gpus: list[int] | int = args.gpus
lambda_R: float = args.lambda_R

# logger
utils.setup_logger(filepath=dir.joinpath("root.log"))
logger = getLogger(__name__)

# Trainer
plugins = [
    # FSDPPrecision("16-mixed")
]
trainer_loggers = [
    TensorBoardLogger(dir, "tb_loggers"),
    CSVLogger(dir, "csv_logger"),
    ImageSave(dir, "image"),
]
trainer = L.Trainer(
    accelerator="gpu",
    strategy="fsdp",
    devices=gpus,
    logger=trainer_loggers,
    benchmark=True,
    plugins=plugins,
)

datamodule = data.Synapse("Abdomen")
model = models.VM(data.SYNAPSE_SHAPE, lambda_R)
trainer.test(
    model=model,
    datamodule=datamodule,
)
