import torch
import torch.nn as nn
import numpy as np
import random
import os

from torchvision import models

import math
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm


class Resnet(nn.Module):
    """RESNET model with BatchNorm replaced with GroupNorm"""

    def __init__(self, num_classes, resnet_size, pretrained=False):
        super().__init__()

        # Retrieve resnet of appropriate size
        resnet = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        assert (
            resnet_size in resnet.keys()
        ), f"Resnet size {resnet_size} is not supported!"

        self._name = f"Resnet{resnet_size}"
        self.backbone = resnet[resnet_size]()

        if pretrained:
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    models.resnet.model_urls[f"resnet{resnet_size}"],
                    progress=True,
                )
            )

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # # Replace batch norm with group norm

    def forward(self, x):
        return self.backbone(x)

    def name(self):
        return self._name