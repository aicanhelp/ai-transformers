from transformers import PreTrainedModel, get_linear_schedule_with_warmup, AdamW
from .trainer_base import *


@configclass()
class TrainerOptimizersConfig:
    learning_rate: float = field(5e-5, "The initial learning rate for Adam.")
    weight_decay: float = field(0.0, "Weight decay if we apply some.")
    adam_epsilon: float = field(1e-8, "Epsilon for Adam optimizer.")
    max_grad_norm: float = field(1.0, "Max gradient norm.")
    warmup_steps: int = field(0, "Linear warmup over warmup_steps.")
    gradient_accumulation_steps: int = field(
        1, "Number of updates steps to accumulate before performing a backward/update pass.")


class TaskTrainerOptimizers():
    def __init__(self, trainer_env: TrainerEnv,
                 model: PreTrainedModel,
                 num_train_steps: int,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LambdaLR = None
                 ):
        self._env = trainer_env
        self.config: TrainerOptimizersConfig = trainer_env.get_config(TrainerOptimizersConfig)
        self.origin_model = model
        self.model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._num_train_steps = num_train_steps

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        if not self._optimizer:
            self._optimizer = self._create_optimizer()
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        if not self._scheduler:
            self._scheduler = self._create_scheduler(self.optimizer, self._num_train_steps)

        return self._scheduler

    def set_for_start_train(self):
        if self._env.config.fp16:
            self.model, self._optimizer = self._env.apex_model_optimizer(self.model, self.optimizer)

            # multi-gpu training (should be after apex fp16 initialization)
        if self._env.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Distributed training (should be after apex fp16 initialization)
        if self._env.config.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self._env.config.local_rank],
                output_device=self._env.config.local_rank,
                find_unused_parameters=True
            )
        self.model.zero_grad()

    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

    def _create_scheduler(self, optimizer, num_training_steps):
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=num_training_steps
        )

    def step_backward(self, inputs: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self._env.device)

        outputs = self.model.forward(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self._env.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        if self._env.config.fp16:
            self._env.fp16_backward(loss, self.optimizer)
        else:
            loss.backward()

        return loss.item()

    def clip_and_step(self):
        if self._env.config.fp16:
            self._env.fp16_clip(self.optimizer, self.config.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        if self._env.is_tpu_available():
            self._env.tpu_step(self.optimizer)
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.model.zero_grad()
