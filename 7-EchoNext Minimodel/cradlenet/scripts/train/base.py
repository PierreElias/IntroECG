import pytorch_lightning as pl
from azureml.core.run import Run
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.profilers import SimpleProfiler
from cradlenet.lightning.logger.logger import get_logger

from typing import List
from argparse import Namespace


def train(
    args: Namespace,
    data_module: pl.LightningDataModule,
    module: pl.LightningModule,
    callbacks: List[pl.Callback],
    gpu: bool = False,
):
    """Basic training script.

    Combines a LightningModule with a LightningDataModule and Callbacks to train a model.
    """
    logger = get_logger()

    trainer = pl.Trainer(
        logger=logger,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        enable_checkpointing=bool(args.ckpt_dir),
        default_root_dir=args.ckpt_dir,
        enable_progress_bar=args.progbar,
        accelerator="gpu" if gpu else "cpu",
        devices=1,
        profiler=SimpleProfiler(),
        precision=32,
    )

    print("created trainer, starting training")
    module = trainer.fit(module, datamodule=data_module)
