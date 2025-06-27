import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from typing import List, Dict


def one_hot_encode(labels, nclasses):
    ohe = np.zeros((len(labels), nclasses), dtype=bool)
    rows = np.arange(len(labels))
    ohe[rows, labels] = True
    return ohe


def classification_metrics(
    preds: List[np.ndarray], labels: List[np.ndarray], multiclass: bool, prefix: str = ""
) -> Dict[str, float]:
    """Supports multiclass, multilabel and binary classification

    multiclass:
        preds shape: (N, N_classes)
        labels shape: (N,)

    binary:
        preds shape: (N,)
        labels shape: (N,)

    multilabel:
        preds shape: (N, Nc)
        labels shape: (N, Nc)

    computes roc_auc_score, average_precision_score, f1_score@0.5
    in multiclass case f1 score is computed using the class with highest predicted value
    """
    assert len(preds) == len(labels)
    if not multiclass:
        assert preds[0].shape == labels[0].shape

    multilabel = len(labels[0].shape) > 1 and not multiclass

    if multiclass:
        preds = np.vstack(preds)
        labels = np.hstack(labels)
        ohe_labels = one_hot_encode(labels, preds.shape[-1])
    elif multilabel:
        preds = np.vstack(preds)
        ohe_labels = np.vstack(labels)
    else:
        preds = np.hstack(preds)
        ohe_labels = np.hstack(labels)

    roc_auc = roc_auc_score(ohe_labels, preds, average=None)
    ap = average_precision_score(ohe_labels, preds, average=None)
    f1 = f1_score(
        labels if multiclass else ohe_labels,
        np.argmax(preds, axis=-1) if multiclass else preds > 0.5,
        average=None,
    )

    if multiclass or multilabel:
        metrics = {f"{prefix}f1_{i}": s for i, s in enumerate(f1)}
        metrics.update({f"{prefix}ap_{i}": s for i, s in enumerate(ap)})
        metrics.update({f"{prefix}rocauc_{i}": s for i, s in enumerate(roc_auc)})
    else:
        metrics = {f"{prefix}f1": f1}
        metrics.update({f"{prefix}ap": ap})
        metrics.update({f"{prefix}rocauc": roc_auc})

    return metrics


class ClassificationMetrics(pl.Callback):
    """Implements metrics for binary and multiclass classification.

    Note:
        For the training set, the model shifts while we accumulate model outputs and the result is an
    estimate of model performance across the dataset and across different states of the model.
    If you want to measure model performance on the training set, you have to specify an additional
    validation dataloader with the train set.

    Produces the following metrics:
        f1_score@0.5, average_precision_score, roc_auc_score
    Assumes
        - labels are the 2nd element in the batch (e.g., batch[1]).
        - logits are returned in a dictionary from the module under key 'logits'
    """

    def __init__(self, binary=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_samples = []
        self.train_labels = []
        self.val_samples = []
        self.val_labels = []
        self.binary = binary

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if trainer.sanity_checking:
            return
        labels = batch[1].detach().cpu().numpy()
        if self.binary:
            preds = torch.sigmoid(outputs["logits"])
        else:
            preds = F.softmax(outputs["logits"], dim=-1)

        self.train_samples.append(preds.detach().cpu().numpy())
        self.train_labels.append(labels)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if trainer.sanity_checking:
            return
        labels = batch[1].detach().cpu().numpy()
        if self.binary:
            preds = torch.sigmoid(outputs["logits"])
        else:
            preds = F.softmax(outputs["logits"], dim=-1)

        self.val_samples.append(preds.detach().cpu().numpy())
        self.val_labels.append(labels)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        metrics = classification_metrics(
            self.train_samples,
            self.train_labels,
            multiclass=not self.binary,
            prefix="train_epoch_",
        )

        logger: MLFlowLogger = trainer.logger
        logger.log_metrics(metrics, trainer.current_epoch)
        self.train_samples = []
        self.train_labels = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        metrics = classification_metrics(
            self.val_samples,
            self.val_labels,
            multiclass=not self.binary,
            prefix="val_epoch_",
        )
        logger: MLFlowLogger = trainer.logger
        logger.log_metrics(metrics, trainer.current_epoch)
        self.val_samples = []
        self.val_labels = []
