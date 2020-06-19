from .trainer_base import *
from .trainer_counter import TaskTrainedCounter
from .trainer_logger import TaskTrainerLogger
from .trainer_optimizers import TaskTrainerOptimizers
from .trainer_saver import TaskTrainerSaver
from .trainer_dataloaders import TaskTrainerDataLoaders
from .trainer_evaluator import TaskTrainerEvaluator


class TaskTrainerContext(TrainerBase):
    def __init__(self, args: TaskTrainingArguments,
                 model: PreTrainedModel,
                 optimizers: TaskTrainerOptimizers,
                 data_loaders: TaskTrainerDataLoaders,
                 trainer_logger: TaskTrainerLogger,
                 evaluator: TaskTrainerEvaluator):
        super().__init__(args)
        self.model = model
        self.optimizers = optimizers if optimizers else TaskTrainerOptimizers(args, model)
        self.data_loaders = data_loaders
        self.evaluator = evaluator if evaluator else TaskTrainerEvaluator.build(args, model, data_loaders)
        self.trainer_logger = trainer_logger if trainer_logger else TaskTrainerLogger(args, model)
        self.saver = TaskTrainerSaver(args, model)
        self.train_counter = None

        self.__init_env()

    def __init_env(self):
        set_seed(self._args.seed)
        # Create output directory if needed
        if self.is_local_master():
            os.makedirs(self._args.output_dir, exist_ok=True)
        if self.is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

    def init_context_for_train(self, model_path):
        self.train_counter = TaskTrainedCounter(model_path, self._args, len(self.data_loaders.get_train_dataloader()))
        self.saver.load_check_point(model_path, self.optimizers)
        self.optimizers.set_for_start_train()


class TaskTrainStep(TrainerBase):
    def __init__(self, args: TaskTrainingArguments, optimizers: TaskTrainerOptimizers, total_steps):
        self._args = args
        self._model = optimizers.get_model()
        self._optimizer = optimizers.get_optimizer()
        self._scheduler = optimizers.get_scheduler()
        self._total_steps = total_steps

    def _step_backward(self, inputs: Dict[str, torch.Tensor]) -> float:
        self._model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self._args.device)

        outputs = self._model.forward(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self._args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self._args.gradient_accumulation_steps > 1:
            loss = loss / self._args.gradient_accumulation_steps

        if self._args.fp16:
            with self.amp.scale_loss(loss, self._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _is_step(self, step):
        return (step + 1) % self._args.gradient_accumulation_steps == 0 or (
            # last step in epoch but step is always smaller than gradient_accumulation_steps
                self._total_steps <= self._args.gradient_accumulation_steps
                and (step + 1) == self._total_steps
        )

    def _clip_and_step(self):
        if self._args.fp16:
            torch.nn.utils.clip_grad_norm_(self.amp.master_params(self._optimizer),
                                           self._args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._args.max_grad_norm)

        if self.is_tpu_available():
            self.xm.optimizer_step(self._optimizer)
        else:
            self._optimizer.step()

        self._scheduler.step()
        self._model.zero_grad()

    def execute(self, step, inputs: Dict[str, torch.Tensor]) -> (float, bool):
        loss = self._step_backward(inputs)

        if not self._is_step(step): return loss, False

        self._clip_and_step()

        return loss, True


class TaskTrainEpoch(TrainerBase):
    def __init__(self, epoch,
                 taskContext: TaskTrainerContext):
        self.task_context = taskContext
        self.epoch_iterator = tqdm(taskContext.data_loaders.get_train_dataloader(), desc="Iteration",
                                   disable=not self.is_local_master())
        self.epoch = epoch
        self.epoch_len = len(self.epoch_iterator)
        self._train_counter = taskContext.train_counter
        self._trainer_step = self.__create_step()

    def __create_step(self):
        return TaskTrainStep(self._args, self.task_context.optimizers, self.epoch_len)

    def execute(self):
        for step, inputs in enumerate(self.epoch_iterator):
            if not self._train_counter.check_step_for_break(self.epoch_iterator):
                break

            if self._train_counter.check_step_for_skip():
                continue

            loss, do_step = self._trainer_step.execute(step, inputs)
            self._process_step(step, do_step, loss)

        self.epoch_iterator.close()
        self.task_context.trainer_logger.log_train_epoch()

    def _process_step(self, step, do_step, loss):
        self._train_counter.on_train_step(step, self.epoch_len, loss)

        if do_step and self.is_local_master():
            self._process_step(self.epoch, self._train_counter.tr_loss,
                               self._train_counter.logging_loss)
            self._train_counter.after_process_step()

    def _process_step_loss(self, epoch, tr_loss, logging_loss):
        if self.task_context.trainer_logger.is_need_log_step(self._train_counter.global_step):
            logs: Dict[str, float] = {}
            logs["loss"] = (tr_loss - logging_loss) / self._train_counter.logging_steps
            logs["learning_rate"] = self.task_context.optimizers.get_scheduler().get_last_lr()[0]

            self.task_context.trainer_logger.log_train_step(epoch, self._train_counter.global_step, logs)

            if self._args.evaluate_during_training:
                self.task_context.evaluator.evaluate(self.task_context.data_loaders.get_eval_dataloader(),
                                                     description="Training setup evaluate, step={}".format(
                                                         self._train_counter.global_step),
                                                     max_steps=self._args.eval_max_steps)

        self.task_context.saver.save_check_point(self.task_context.optimizers, self._train_counter.global_step)


class TaskTrainLoop(TrainerBase):
    def __init__(self, taskContext: TaskTrainerContext):
        self.task_context = taskContext
        self._train_counter = taskContext.train_counter

    def execute(self):
        train_iterator = trange(
            self._train_counter.epochs_trained, int(self._train_counter.num_train_epochs), desc="Epoch",
            disable=not self.is_local_master()
        )

        for epoch in train_iterator:
            if not self._train_counter.check_step_for_break(train_iterator):
                break
            TaskTrainEpoch(epoch, self.task_context).execute()

        train_iterator.close()
        self.task_context.trainer_logger.log_train_end()
        return self._train_counter.task_train_output()


class TaskTrainer():
    def __init__(self, context: TaskTrainerContext):
        self._context = context

    @staticmethod
    def build(args: TaskTrainingArguments,
              model: PreTrainedModel,
              data_loaders: TaskTrainerDataLoaders,
              tb_writer: Optional["SummaryWriter"] = None,
              compute_metrics: Optional[Callable[[TaskEvalPrediction], Dict]] = None):
        TaskTrainer(TaskTrainerContext(args=args,
                                       model=model,
                                       optimizers=TaskTrainerOptimizers(args, model),
                                       data_loaders=data_loaders,
                                       trainer_logger=TaskTrainerLogger(args, model, tb_writer),
                                       evaluator=TaskTrainerEvaluator.build(args, model, data_loaders, compute_metrics)
                                       ))

    def train(self, model_path: Optional[str] = None):
        self._context.init_context_for_train(model_path)

        self._context.trainer_logger.log_pre_train(self._context.data_loaders.get_train_dataloader(),
                                                   self._context.train_counter)

        return TaskTrainLoop(self._context).execute()
