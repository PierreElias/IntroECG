import torch
from torch.nn import Module
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch import Tensor
from typing import Optional

import torch.nn as nn


class ClassificationModule(LightningModule):
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

    def __init__(
        self, model: Module, lr: float, binary: bool = True, pos_weight: Optional[Tensor] = None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.binary = binary
        self.pos_weight = (
            nn.Parameter(pos_weight, requires_grad=False) if pos_weight is not None else pos_weight
        )

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
            loss = F.cross_entropy(logits, loss, pos_weight=self.pos_weight)
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
            loss = F.cross_entropy(logits, labels, pos_weight=self.pos_weight)
        return {"loss": loss, "logits": logits.detach()}

    def predict_step(self, batch, predict_step):
        return {"probs": self.forward(batch)}
