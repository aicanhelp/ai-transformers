from torch.utils.data import Dataset
from typing import List, Optional

from .data_converter import TaskDataConverter
from .data_processor import TaskDataProcessor

from .data_store import TaskDataStore, TaskDataset
from ..train.trainer import torch_distributed_zero_first


class TaskDatasetFactory:
    def __init__(self, data_store: TaskDataStore,
                 processor: TaskDataProcessor,
                 dataConverter: TaskDataConverter):
        self._data_store = data_store
        self._processor = processor
        self._converter = dataConverter

    def _create_dataset(self, evaluate=False, limit_length: Optional[int] = None, local_rank=-1) -> Dataset:
        with torch_distributed_zero_first(local_rank):
            dataset = self._data_store.load_dataset(limit_length, evaluate)
            if not dataset:
                features = self.__generate_features(limit_length, evaluate)
                dataset = TaskDataset(features)
                if local_rank in [-1, 0]:
                    self._data_store.save_dataset(dataset, evaluate)
            return dataset

    def create_train_dataset(self, limit_length: Optional[int] = None, local_rank=-1) -> Dataset:
        return self._create_dataset(True, limit_length, local_rank)

    def create_eval_dataset(self, limit_length: Optional[int] = None, local_rank=-1) -> Dataset:
        return self._create_dataset(True, limit_length, local_rank)

    def create_predict_dataset(self, limit_length: Optional[int] = None, local_rank=-1) -> Dataset:
        with torch_distributed_zero_first(local_rank):
            features = self.__generate_features(limit_length, True)
            return TaskDataset(features)

    def __generate_features(self, limit_length: Optional[int] = None, evaluate=False):
        examples = (
            self._processor.get_eval_examples(limit_length) if evaluate
            else self._processor.get_train_examples(limit_length)
        )

        return self._converter.convert(examples)
