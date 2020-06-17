import torch
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, AdamW

from transformersx.train.training_args import TaskTrainingArguments
from .trainer_base import TrainerBase


class TaskTrainerOptimizers(TrainerBase):
    def __init__(self, args: TaskTrainingArguments,
                 model: PreTrainedModel,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LambdaLR = None):
        super().__init__(args)
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

    def get_model(self) -> PreTrainedModel:
        return self._model

    def get_optimizer(self) -> torch.optim.Optimizer:
        if not self._optimizer:
            self._optimizer = self._create_optimizer()
        return self._optimizer

    def get_scheduler(self, num_training_steps=-1) -> torch.optim.lr_scheduler.LambdaLR:
        if not self._scheduler:
            self._scheduler = self._create_scheduler(self.get_optimizer(), num_training_steps)

        return self._scheduler

    def set_for_start_train(self):
        self._model.to(self._args.device)
        if self._args.fp16:
            if not self.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self._model, self._optimizer = self.amp.initialize(self._model, self._optimizer,
                                                               opt_level=self._args.fp16_opt_level)

            # multi-gpu training (should be after apex fp16 initialization)
        if self._args.n_gpu > 1:
            self._model = torch.nn.DataParallel(self._model)

        # Distributed training (should be after apex fp16 initialization)
        if self._args.local_rank != -1:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model,
                device_ids=[self._args.local_rank],
                output_device=self._args.local_rank,
                find_unused_parameters=True,
            )
        self._model.zero_grad()

    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self._args.weight_decay,
            },
            {
                "params": [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self._args.learning_rate, eps=self._args.adam_epsilon)

    def _create_scheduler(self, optimizer, num_training_steps):
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self._args.warmup_steps, num_training_steps=num_training_steps
        )
