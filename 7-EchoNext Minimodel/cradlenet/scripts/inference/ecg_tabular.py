import argparse
from pathlib import Path

import torch
from pytorch_lightning.callbacks import EarlyStopping

from cradlenet.lightning.callbacks.loss_logger import LossLogger
from cradlenet.lightning.callbacks.metrics.classification_metrics import ClassificationMetrics

from cradlenet.datasets.numpy import NumpyDataset
from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular

from cradlenet.lightning.datamodules.base import BaseDataModule
from cradlenet.lightning.modules.resnet1d_with_tabular import Resnet1dWithTabularModule
from cradlenet.lightning.modules.classification import ClassificationModule

from cradlenet.scripts.inference.base import inference

from cradlenet.lightning.callbacks.output_writer import OutputWriter


def parse_args():
    parser = argparse.ArgumentParser(description="train script")
    parser.add_argument("--features_path", type=Path, help="path to train feature data")
    parser.add_argument("--labels_path", type=Path, help="path to train label data")
    parser.add_argument("--tabular_path", type=Path, help="path to train tabular data")

    parser.add_argument(
        "--legacy",
        default=None,
        type=str,
        help="legacy model checkpoint. one of [valvenet, cadnet]",
    )
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--num_workers", default=6, type=int, help="dataloader workers")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint path")
    parser.add_argument(
        "--eval", action="store_true", help="run inference with labels and calculate metrics"
    )
    parser.add_argument("--progbar", action="store_true", help="enable progbar")
    parser.add_argument("--write_outputs", action="store_true", help="write model outputs")
    parser.add_argument(
        "--output_dir", default=".", type=Path, help="directory to write outputs to"
    )
    parser.add_argument(
        "--binary", action="store_true", help="use binary or multilabel metrics / loss"
    )
    parser = Resnet1dWithTabularModule.add_model_specific_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    print("cuda is available: ", torch.cuda.is_available())

    args = parse_args()
    print(f"parsed arguments: {args}")

    if args.eval:
        data = NumpyDataset((args.features_path, args.labels_path, args.tabular_path))
        data_module = BaseDataModule(
            val_dataset=data, num_workers=args.num_workers, batch_size=args.batch_size
        )
    else:
        data = NumpyDataset((args.features_path, args.tabular_path))
        data_module = BaseDataModule(pred_dataset=data, num_workers=args.num_workers)
    print(f"created data module: {data_module}")

    if args.legacy:
        if args.legacy == "valvenet":
            model_kwargs = {"len_tabular_feature_vector": 7, "filter_size": 64}
        elif args.legacy == "cadnet":
            model_kwargs = {"len_tabular_feature_vector": 14, "filter_size": 32}
        elif args.legacy == "echonext":
            model_kwargs = {
                "len_tabular_feature_vector": args.len_tabular_feature_vector,
                "filter_size": args.filter_size,
                "num_classes": args.num_classes,
            }
        else:
            raise ValueError(
                f"legacy argument must be valvenet, cadnet or echonext not {args.legacy}"
            )
        module = Resnet1dWithTabularModule(model_kwargs=model_kwargs, lr=0, binary=args.binary)
        weights = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        module.model.load_state_dict(weights["model"])
    else:
        module = Resnet1dWithTabularModule.load_from_checkpoint(
            args.checkpoint,
            map_location=torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
        )

    print(f"created lightning module: {module}")

    if args.eval:
        callbacks = [
            ClassificationMetrics(binary=args.binary),
        ]
    else:
        callbacks = []

    if args.write_outputs:
        callbacks.append(OutputWriter(output_dir=args.output_dir))

    print(f"created callbacks: {tuple(map(type, callbacks))}")

    inference(args, data_module, module, callbacks, gpu=torch.cuda.is_available())
    print("Finished.")
