import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from pathlib import Path

import numpy as np


class OutputWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path):
        super().__init__("epoch")
        self.output_dir = output_dir
        self.val_results = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.val_results.append({k: v.detach().cpu() for k, v in outputs.items() if v.ndim > 0})

    def on_validation_epoch_end(self, trainer, pl_module):
        d = {
            k: torch.cat(tuple(x[k] for x in self.val_results)).numpy()
            for k in next(iter(self.val_results))
        }

        output_dir = self.output_dir / "validation_loop"
        output_dir.mkdir(parents=True, exist_ok=True)

        for k in d:
            np.save(output_dir / f"{k}.npy", d[k])
        self.val_results = []

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        d = {
            k: torch.cat(tuple(x[k] for x in predictions)).numpy() for k in next(iter(predictions))
        }
        output_dir = self.output_dir / "prediction_loop"
        output_dir.mkdir(exist_ok=True, parents=True)
        for k in d:
            np.save(output_dir / f"{k}.npy", d[k])
