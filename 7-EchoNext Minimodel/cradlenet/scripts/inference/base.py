import pytorch_lightning as pl
from azureml.core.run import Run
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils.data import DataLoader

from typing import List
from argparse import Namespace
from cradlenet.lightning.logger.logger import get_logger


def inference(
    args: Namespace,
    dataloader: DataLoader,
    module: pl.LightningModule,
    callbacks: List[pl.Callback],
    gpu: bool = False,
):
    """Basic eval script.

    Combines a LightningModule with a LightningDataModule and Callbacks to run a model through validation.
    """
    logger = get_logger()

    logger.log_hyperparams(args)
    print("logging hyperparams")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=args.progbar,
        accelerator="gpu" if gpu else "cpu",
        devices=1,
        profiler=SimpleProfiler(),
        precision=32,
    )

    if args.eval:
        print(f"running validation on {dataloader}")
        trainer.validate(module, dataloaders=dataloader)
    else:
        print(f"running prediction on {dataloader}")
        trainer.predict(module, dataloaders=dataloader)
