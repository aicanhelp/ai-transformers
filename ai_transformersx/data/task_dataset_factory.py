from torch.utils.data import Dataset
from transformers import torch_distributed_zero_first
from typing import List, Optional

from .task_data_converter import TaskDataConverter
from .task_data_processor import TaskDataProcessor

from .task_data_store import TaskDataStore, TaskDataset


class TaskDatasetGenerator:
    def __init__(self, data_store: TaskDataStore, processor: TaskDataProcessor,
                 dataConverter: TaskDataConverter):
        self._data_store = data_store
        self._processor = processor
        self._converter = dataConverter

    def dataset(self, predict=False, limit_length: Optional[int] = None, evaluate=False, local_rank=-1) -> Dataset:
        with torch_distributed_zero_first(local_rank):
            dataset = self._data_store.load_dataset(limit_length, evaluate) if not predict else None
            if not dataset:
                features = self.__generate_features(limit_length, evaluate)
                dataset = TaskDataset(features)
                if not predict and local_rank in [-1, 0]:
                    self._data_store.save_dataset(dataset, evaluate)
            return dataset

    def __generate_features(self, limit_length: Optional[int] = None, evaluate=False):
        examples = (
            self._processor.get_eval_examples(limit_length) if evaluate
            else self._processor.get_train_examples(limit_length)
        )

        return self._converter.convert(examples)
