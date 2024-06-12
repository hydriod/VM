from .logger import setup_logger
import torch

EPS = torch.finfo(torch.float32).eps
