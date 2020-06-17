import time
from typing import Optional, List

import torch

from ai_transformersx import InputFeatures
from torch.utils.data import Dataset
import os

from .data_args import TaskDataArguments
from ..transformersx_base import log, join_path


class TaskDataset(Dataset):
    def __init__(self, features: List[InputFeatures]):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class TaskDataStore():
    dataStoreId = None
    dataArgs = None

    def load_dataset(self, limit_length: Optional[int] = None, evaluate=False) -> TaskDataset:
        raise NotImplementedError()

    def save_dataset(self, dataset: TaskDataset, evaluate=False):
        raise NotImplementedError()


class LocalDataStore(TaskDataStore):
    def __init__(self, dataStoreId, dataArgs: TaskDataArguments):
        self.dataStoreId = dataStoreId
        self.dataArgs = dataArgs

    def load_dataset(self, limit_length: Optional[int] = None, evaluate=False) -> TaskDataset:
        saved_file = self._create_save_file(evaluate)
        if not os.path.exists(saved_file):
            return None
        log.info("Loading dataset from file {}".format(saved_file))
        start = time.time()
        features = torch.load(saved_file)
        log.info(
            f"Loaded dataset from file %s [took %.3f s]", saved_file, time.time() - start
        )
        return TaskDataset(features)

    def _create_save_file(self, evaluate):
        saved_file = "{}_{}".format(self.dataStoreId, ("eval" if evaluate else "train"))
        return join_path(self.dataArgs.data_dir, saved_file)

    def save_dataset(self, dataset: TaskDataset, evaluate=False):
        if not dataset:
            return
        saved_file = self._create_save_file(evaluate)
        if not os.path.exists(saved_file) or self.dataArgs.overwrite:
            log.info("Saving dataset into  file {}".format(saved_file))
            start = time.time()
            torch.save(dataset.features, saved_file)

            log.info(
                f"Saved dataset into cached file %s [took %.3f s]", saved_file, time.time() - start
            )
