from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    """Takes any number of paths to numpy files and loads them in a dataset.

    The NumpyDataset object will load the numpy arrays and check their dimensions. It is assumed they are
        sampled on the first axis of any array. That dimension must be equal across the arrays.

    indexing the dataset object will return a tuple of indexed numpy arrays. The order of the arrays is determined
        by the order the paths were passed during construction.

    For example:
        paths = ('train_features.npy', 'train_labels.npy')
        dataset = NumpyDataset(paths)
        dataset[i] = (np.load('train_features.npy')[i], np.load('train_labels.npy')[i])

    The dataset class also supports transforms on the data in the form of an iterable of callable functions.
        These transforms are run in the order they are passed to the dataset object.
    """

    def __init__(
        self, items: Iterable[Path], transforms: Optional[Iterable[Callable]] = None, mmap_mode="r"
    ):
        super().__init__()
        assert 3 >= len(items) >= 1
        self.items = tuple(np.load(x, mmap_mode=mmap_mode, allow_pickle=True) for x in items)
        lens = set(map(len, self.items))
        assert len(lens) == 1

        self.transforms = transforms or []

    def __len__(self) -> int:
        return len(self.items[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        ret = tuple(torch.tensor(x[idx]).to(dtype=torch.float32) for x in self.items)

        for fn in self.transforms:
            ret = fn(ret)

        return ret
