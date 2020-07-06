from transformers import DataCollator

from .trainer_base import *
from ..data import TaskDatasetFactory


@configclass
class TrainerDataloadersConfig():
    pass


class TaskTrainerDataLoaders():
    def __init__(self, trainer_env: TrainerEnv,
                 config: TrainerDataloadersConfig,
                 dataset_factory: TaskDatasetFactory,
                 data_collator: DataCollator):
        self._env = trainer_env
        self.config = config
        self._data_collator = data_collator
        self._dataset_factory = dataset_factory
        self._local_rank = trainer_env.config.local_rank
        self.train_batch_size = trainer_env.batch_train_size()
        self.eval_batch_size = trainer_env.batch_eval_size()

    def _create_dataloader(self, dataset, sampler, batch_size):
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False if sampler is not None else True,
            collate_fn=self._data_collator,
        )
        if self._env.is_tpu_available():
            data_loader = self._env.get_tpu_dataloader(data_loader)
        return data_loader

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self._dataset_factory.create_train_dataset(local_rank=self._local_rank) \
            if self._dataset_factory else None
        assert train_dataset, '"Trainer: training requires a train_dataset."'
        if self._env.is_tpu_available():
            train_sampler = self._env.get_tpu_sampler(train_dataset)
        else:
            train_sampler = (
                RandomSampler(train_dataset)
                if self._local_rank == -1
                else DistributedSampler(train_dataset)
            )

        return self._create_dataloader(train_dataset, train_sampler, self.train_batch_size)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_dataset = self._dataset_factory.create_eval_dataset(
            local_rank=self._local_rank) if not eval_dataset and self._dataset_factory else eval_dataset
        assert eval_dataset, "Trainer: evaluation requires an eval_dataset."

        sampler = self._env.get_tpu_sampler(eval_dataset) if self._env.is_tpu_available() else None

        return self._create_dataloader(eval_dataset, sampler, self.eval_batch_size)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        sampler = None
        if self._env.is_tpu_available():
            sampler = self._env.get_tpu_sampler(test_dataset) if self._env.is_tpu_available() else None

        return self._create_dataloader(test_dataset, sampler, self.eval_batch_size)
