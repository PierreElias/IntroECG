from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from typing import Optional
from torch.utils.data import Dataset


class BaseDataModule(LightningDataModule):
    """Implements a basic lightning data module class.

    This class sets up dataloaders for training and validation.

    Validation data is sequentially loaded and training data is randomly sampled.
    """

    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        pred_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        val_batch_size: Optional[int] = None,
        replacement: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.prepare_data_per_node = False
        self.num_workers = num_workers
        self.replacement = replacement
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.pred_dataset = pred_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        assert self.train_dataset
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                self.train_dataset,
                replacement=self.replacement,
                num_samples=len(self.train_dataset),
            ),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        assert self.val_dataset
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            sampler=SequentialSampler(self.val_dataset),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        assert self.test_dataset
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            sampler=SequentialSampler(self.test_dataset),
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        assert self.pred_dataset
        return DataLoader(
            self.pred_dataset,
            batch_size=self.val_batch_size,
            sampler=SequentialSampler(self.pred_dataset),
            num_workers=self.num_workers,
        )
