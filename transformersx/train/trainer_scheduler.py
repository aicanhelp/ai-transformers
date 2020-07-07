from .trainer_base import *


@configclass()
class TrainerSchedulerConfig:
    max_steps: int = field(
        -1, "If > 0: set total number of training steps to perform. Override num_train_epochs.")
    num_train_epochs: float = field(3.0, "Total number of training epochs to perform.")

    # not field
    train_data_len = 0
    gradient_accumulation_steps = 1
    train_batch_size = 16
    global_step = 0


@configclass()
class TaskTrainedScheduler():
    epoch = 0
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss = 0.0
    logging_loss = 0.0

    total_steps = 0
    num_train_epochs = 0
    total_train_batch_size = 0

    def __init__(self, trainer_env: TrainerEnv, config: TrainerSchedulerConfig):
        self._env = trainer_env
        self.config = config
        self.global_step = config.global_step

        self._cal_epochs()
        self._cal_steps()

    def _cal_epochs(self):
        if self.config.max_steps > 0:
            self.t_total = self.config.max_steps
            self.num_train_epochs = (
                    self.config.max_steps // (self.config.train_data_len // self.config.gradient_accumulation_steps) + 1
            )
        else:
            self.t_total = int(
                self.config.train_data_len // self.config.gradient_accumulation_steps * self.config.num_train_epochs)
            self.num_train_epochs = self.config.num_train_epochs

        if self._env.is_tpu_available():
            self.total_train_batch_size = self.config.train_batch_size * self._env.xrt_world_size()
        else:
            self.total_train_batch_size = (
                    self.config.train_batch_size
                    * self.config.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self._env.config.local_rank != -1 else 1)
            )

    def _cal_steps(self):
        if self.global_step == 0:
            return
        try:
            self.epochs_trained = self.global_step // (
                    self.config.train_data_len // self.config.gradient_accumulation_steps)
            self.steps_trained_in_current_epoch = self.global_step % (
                    self.config.train_data_len // self.config.gradient_accumulation_steps
            )

            log.info("  Continuing training from checkpoint, will skip to saved global_step")
            log.info("  Continuing training from epoch %d", self.epochs_trained)
            log.info("  Continuing training from global step %d", self.global_step)
            log.info("  Will skip the first %d steps in the first epoch", self.steps_trained_in_current_epoch)
        except ValueError:
            self.global_step = 0

    def on_train_step(self, step, epoch_len, loss):
        self.global_step += 1
        self.epoch = self.epoch + (step + 1) / epoch_len
        self.tr_loss += loss

    def after_process_step(self):
        self.logging_loss = self.tr_loss

    def check_step_for_break(self, iterator):
        if self.total_steps <= 0 or self.global_step < self.total_steps:
            return True
        iterator.close()
        return False

    def check_step_for_skip(self):
        if self.steps_trained_in_current_epoch <= 0:
            return False
        self.steps_trained_in_current_epoch -= 1
        return True

    def task_train_output(self):
        TaskTrainOutput(self.global_step, self.tr_loss / self.global_step)

    def is_step(self, step, total_steps):
        return (step + 1) % self.config.gradient_accumulation_steps == 0 or (
            # last step in epoch but step is always smaller than gradient_accumulation_steps
                total_steps <= self.config.gradient_accumulation_steps
                and (step + 1) == total_steps
        )
