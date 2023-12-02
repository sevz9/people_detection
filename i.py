import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import lightning as L
import matplotlib.pyplot as plt
import torch
import torchmetrics
import xmltodict
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS, STEP_OUTPUT,
                                               TRAIN_DATALOADERS)
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large)
from torchvision.utils import draw_bounding_boxes
