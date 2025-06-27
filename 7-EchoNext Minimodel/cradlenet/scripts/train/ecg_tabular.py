import argparse
from pathlib import Path
import torch
from pytorch_lightning.callbacks import EarlyStopping

from cradlenet.lightning.callbacks.loss_logger import LossLogger
from cradlenet.lightning.callbacks.metrics.classification_metrics import ClassificationMetrics

from cradlenet.datasets.numpy import NumpyDataset

from cradlenet.lightning.datamodules.base import BaseDataModule
from cradlenet.lightning.modules.resnet1d_with_tabular import Resnet1dWithTabularModule
from cradlenet.scripts.train.base import train
import inspect
import numpy as np
from argparse import Namespace


def parse_args():
    parser = argparse.ArgumentParser(description="train script")
    parser.add_argument("--train_features_path", type=Path, help="path to train feature data")
    parser.add_argument("--train_labels_path", type=Path, help="path to train label data")
    parser.add_argument("--train_tabular_path", type=Path, help="path to train tabular data")

    parser.add_argument("--val_features_path", type=Path, help="path to val feature data")
    parser.add_argument("--val_labels_path", type=Path, help="path to val label data")
    parser.add_argument("--val_tabular_path", type=Path, help="path to val tabular data")

    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "--pos_weight", action="store_true", help="calculate positive label weights for loss"
    )
    parser.add_argument("--binary", action="store_true", help="binary classification or not")
    parser.add_argument("--min_epochs", default=1, type=int)
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--log_every_n_steps", default=1, type=int)
    parser.add_argument("--ckpt_dir", default=None, type=Path)
    parser.add_argument("--progbar", action="store_true", help="display progbar")
    parser.add_argument("--mmap_mode", default=None, type=str)
    parser.add_argument("--num_workers", default=2, type=int)

    parser = Resnet1dWithTabularModule.add_model_specific_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    print("cuda is available: ", torch.cuda.is_available())

    args: Namespace = parse_args()
    print(f"parsed arguments: {args}")

    model_kwargs = {
        k: getattr(args, k)
        for k in vars(args).keys()
        if k in Resnet1dWithTabularModule.MODEL_KWARGS
    }
    print(f"parsed model_kwargs: {model_kwargs}")

    train_data = NumpyDataset(
        (args.train_features_path, args.train_labels_path, args.train_tabular_path),
        mmap_mode=args.mmap_mode,
    )
    val_data = NumpyDataset(
        (args.val_features_path, args.val_labels_path, args.val_tabular_path),
        mmap_mode=args.mmap_mode,
    )

    data_module = BaseDataModule(
        train_data, val_data, num_workers=args.num_workers, batch_size=args.batch_size
    )
    print(f"created data module: {data_module}")

    if args.pos_weight:
        print(f"calculating pos weights with file {args.train_labels_path}")
        labels = np.load(args.train_labels_path)
        pos_weight = torch.tensor(
            (labels.shape[0] - labels.sum(axis=0)) / (labels.sum(axis=0) + 1e-6)
        )
        print(f"pos_weight: {pos_weight}")
        del labels
    else:
        pos_weight = None

    module = Resnet1dWithTabularModule(
        model_kwargs=model_kwargs,
        lr=args.lr,
        binary=args.binary,
        pos_weight=pos_weight,
    )

    print(f"created lightning module: {module}")

    callbacks = [
        LossLogger(),
        EarlyStopping(
            monitor="val_loss", mode="min", patience=20, check_finite=True, verbose=True
        ),
        ClassificationMetrics(binary=args.binary),
    ]
    print(f"created callbacks: {tuple(map(type, callbacks))}")

    train(args, data_module, module, callbacks, gpu=torch.cuda.is_available())
    print("Finished.")
