from .trainer_base import *


@dataclass()
class TaskTrainedCounter(TrainerBase):
    epoch = 0
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss = 0.0
    logging_loss = 0.0

    total_steps = 0
    num_train_epochs = 0
    total_train_batch_size = 0

    def __init__(self, model_path, args: TaskTrainingArguments, train_data_len):
        super().__init__(args)
        self._args = args
        self._model_path = model_path
        self._train_data_len = train_data_len

        self._cal_step_from_checkpoint()
        self._cal_epochs()

    def _cal_epochs(self):
        if self._args.max_steps > 0:
            self.t_total = self._args.max_steps
            self.num_train_epochs = (
                    self._args.max_steps // (self._train_data_len // self._args.gradient_accumulation_steps) + 1
            )
        else:
            self.t_total = int(
                self._train_data_len // self._args.gradient_accumulation_steps * self._args.num_train_epochs)
            self.num_train_epochs = self._args.num_train_epochs

        if self.is_tpu_available():
            self.total_train_batch_size = self.train_batch_size * self.xm.xrt_world_size()
        else:
            self.total_train_batch_size = (
                    self.train_batch_size
                    * self._args.gradient_accumulation_steps
                    * (torch.distributed.get_world_size() if self._args.local_rank != -1 else 1)
            )

    def _cal_step_from_checkpoint(self):
        if self._model_path is not None:
            return
        try:
            self.global_step = int(self._model_path.split("-")[-1].split("/")[0])
            self.epochs_trained = self.global_step // (self._train_data_len // self._args.gradient_accumulation_steps)
            self.steps_trained_in_current_epoch = self.global_step % (
                    self._train_data_len // self._args.gradient_accumulation_steps
            )

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", self.epochs_trained)
            logger.info("  Continuing training from global step %d", self.global_step)
            logger.info("  Will skip the first %d steps in the first epoch", self.steps_trained_in_current_epoch)
        except ValueError:
            self.global_step = 0
            logger.info("  Starting fine-tuning.")

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
