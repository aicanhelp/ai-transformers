from .trainer_base import *

from transformersx.train.training_args import TaskTrainingArguments


class TaskTrainerDataLoaders(TrainerBase):
    def __init__(self, args: TaskTrainingArguments,
                 data_collator: DataCollator,
                 train_dataset: Dataset = None,
                 eval_dataset: Dataset = None):
        super().__init__(args)
        self._data_collator = data_collator
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset

    def _create_dataloader(self, dataset, sampler, batch_size):
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False if sampler is not None else True,
            collate_fn=self._data_collator.collate_batch,
        )
        if self.is_tpu_available():
            data_loader = self.pl.ParallelLoader(data_loader, [self._args.device]).per_device_loader(self._args.device)
        return data_loader

    def get_train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.is_tpu_available():
            train_sampler = self.get_tpu_sampler(self._train_dataset)
        else:
            train_sampler = (
                RandomSampler(self._train_dataset)
                if self._args.local_rank == -1
                else DistributedSampler(self._train_dataset)
            )

        return self._create_dataloader(self._train_dataset, train_sampler, self.train_batch_size)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self._eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self._eval_dataset

        sampler = self.get_tpu_sampler(eval_dataset) if self.is_tpu_available() else None

        return self._create_dataloader(eval_dataset, sampler, self.eval_batch_size)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        sampler = None
        if self.is_tpu_available():
            sampler = self.get_tpu_sampler(test_dataset) if self.is_tpu_available() else None

        return self._create_dataloader(test_dataset, sampler, self.eval_batch_size)
