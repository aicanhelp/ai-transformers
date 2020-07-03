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
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LambdaLR = None):
        self._env = trainer_env
        self.config: TrainerOptimizersConfig = trainer_env.get_config(TrainerOptimizersConfig)
        self._origin_model = model
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

    def get_origin_model(self) -> PreTrainedModel:
        return self._origin_model

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
        if self._env.config.fp16:
            self._model, self._optimizer = self._env.apex_model_optimizer(self._model, self._optimizer)

            # multi-gpu training (should be after apex fp16 initialization)
        if self._env.n_gpu > 1:
            self._model = torch.nn.DataParallel(self._model)

        # Distributed training (should be after apex fp16 initialization)
        if self._env.config.local_rank != -1:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model,
                device_ids=[self._env.config.local_rank],
                output_device=self._env.config.local_rank,
                find_unused_parameters=True
            )
        self._model.zero_grad()

    def _create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

    def _create_scheduler(self, optimizer, num_training_steps):
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=num_training_steps
        )

    def step_backward(self, inputs: Dict[str, torch.Tensor]) -> float:
        self._model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self._env.args.device)

        outputs = self._model.forward(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self._env.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps

        if self._env.args.fp16:
            self._env.fp16_backward(loss, self.get_optimizer())
        else:
            loss.backward()

        return loss.item()

    def clip_and_step(self):
        if self._env.args.fp16:
            self._env.fp16_clip(self.get_optimizer(), self.config.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.max_grad_norm)

        if self._env.is_tpu_available():
            self._env.tpu_step(self.get_optimizer())
        else:
            self._optimizer.step()

        self._scheduler.step()
        self._model.zero_grad()
