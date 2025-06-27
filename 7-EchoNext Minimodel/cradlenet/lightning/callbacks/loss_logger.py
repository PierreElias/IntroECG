from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class LossLogger(pl.Callback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        self.log("train_loss", outputs["loss"], on_step=False, on_epoch=True)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        self.log("val_loss", outputs["loss"], on_step=False, on_epoch=True)
