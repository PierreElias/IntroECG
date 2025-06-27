import torch
from torch.nn import Module
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch import Tensor
from typing import Optional

import torch.nn as nn
from typing import Dict

from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular
import argparse
import inspect


class Resnet1dWithTabularModule(LightningModule):
    """Supports binary and multiclass classification models.

    Notes on batches:
        Module supports operating on data in different ways. Dataloaders with
            tabular data (i.e., another element in the batch) should have it
            at the end.

        training and validation:
            - (features, labels)
            - (features, labels, tabular)
        predict:
            - (features,)
            - (features, tabular)
    """

    PARSER_GROUP_NAME = "Resnet1dWithTabular"
    MODEL_KWARGS = inspect.getfullargspec(ResNet1dWithTabular).args

    def __init__(
        self,
        model_kwargs: Dict,
        lr: float,
        binary: bool = True,
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet1dWithTabular(**model_kwargs)
        self.lr = lr
        self.binary = binary
        self.pos_weight = (
            nn.Parameter(pos_weight, requires_grad=False) if pos_weight is not None else pos_weight
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group(cls.PARSER_GROUP_NAME)
        parser.add_argument("--dropout", default=0.0, type=float, help="dropout fraction")
        parser.add_argument("--filter_size", default=16, type=int, help="filter size")
        parser.add_argument("--conv1_kernel_size", default=15, type=int, help="conv1_kernel_size")
        parser.add_argument(
            "--num_classes", type=int, default=1, help="number of classes on model"
        )
        parser.add_argument(
            "--input_channels", default=12, type=int, help="number of input channels to network"
        )
        parser.add_argument(
            "--len_tabular_feature_vector",
            default=7,
            type=int,
            help="number of input channels to network",
        )
        return parent_parser

    def configure_optimizers(self):
        return Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return torch.sigmoid(logits) if self.binary else torch.softmax(logits)

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            features = (batch[0], batch[2])
        else:
            features = batch[0]
        labels = batch[1]

        logits = self.model(features)

        if self.binary:
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        else:
            loss = F.cross_entropy(logits, loss, weight=self.pos_weight)
        return {"loss": loss, "logits": logits.detach()}

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            features = (batch[0], batch[2])
        else:
            features = batch[0]
        labels = batch[1]

        logits = self.model(features)
        if self.binary:
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        else:
            loss = F.cross_entropy(logits, labels, weight=self.pos_weight)
        return {"loss": loss, "logits": logits.detach()}

    def predict_step(self, batch, predict_step):
        return {"probs": self.forward(batch)}
