from tqdm import tqdm, trange

from .trainer_base import *
from .trainer_scheduler import TaskTrainedScheduler
from .trainer_logger import TaskTrainerLogger
from .trainer_optimizers import TaskTrainerOptimizers
from .trainer_checkpoint import TaskTrainerCheckpoint, TaskTrainerCheckpointer
from .trainer_dataloaders import TaskTrainerDataLoaders
from .trainer_evaluator import TaskTrainerEvaluator, TaskTrainerPredictor
from ..utils import set_seed
import os
from ..model import TaskModelFactory, TaskModel
from ..data import TaskDatasetFactory, TaskDataConverter
from ..transformersx_base import field, configclass


@configclass
class TaskContext:
    task_name: str = field('default')
    data_collator: DataCollator = field(TaskDefaultDataCollatorx())
    compute_metrics = field(default_compute_metrics)
    model_class = field(None)
    data_processor = field(None)


class TaskTrainerBuildContext:
    def task_context(self) -> TaskContext:
        raise NotImplementedError()

    def task_model_factory(self) -> TaskModelFactory:
        raise NotImplementedError()

    def task_dataset_factory(self) -> TaskDatasetFactory:
        raise NotImplementedError()

    def task_data_converter(self) -> TaskDataConverter:
        raise NotImplementedError()


class TaskTrainerContext():
    def __init__(self, train_env: TrainerEnv,
                 build_context: TaskTrainerBuildContext, for_train=True):
        self.env = train_env
        self.task_context = build_context.task_context()
        self.task_model_factory = build_context.task_model_factory()
        self.dataset_factory = build_context.task_dataset_factory()
        self.task_name = self.task_context.task_name

        self.data_loaders = TaskTrainerDataLoaders(self.env, self.dataset_factory,
                                                   self.task_context.data_collator)
        self.trainer_checkpointer = TaskTrainerCheckpointer(self.task_name, self.env)
        self.model, self.optimizers, self.train_counter = self.__create_model_optimizers_counter(for_train)

        self.predictor = TaskTrainerPredictor(self.env, self.model, self.data_loaders)
        self.evaluator = TaskTrainerEvaluator(self.env, self.predictor, self.task_context.compute_metrics)
        self.trainer_logger = TaskTrainerLogger(self.env.args, self.model)

    def __create_model_optimizers_counter(self, for_train=True):
        latest_checkpoint = self.trainer_checkpointer.find_latest_checkpoint()
        model_path = latest_checkpoint.checkpoint_dir if latest_checkpoint else None

        task_model = self.task_model_factory.pretrained_model(model_path).model
        self.__init_model_env(task_model)

        optimizers = TaskTrainerOptimizers(self.env.args, task_model)
        train_scheduler = TaskTrainedScheduler(self.env,
                                               len(self.data_loaders.get_train_dataloader()),
                                               optimizers.config.gradient_accumulation_steps,
                                               self.data_loaders.train_batch_size,
                                               0 if not latest_checkpoint else latest_checkpoint.checkpoint_number()
                                               )
        if for_train:
            self.trainer_checkpointer.init_for_train(task_model, optimizers)
            self.optimizers.set_for_start_train()
        return task_model, optimizers, train_scheduler

    def __init_model_env(self, model):
        model.to(self.env.args.device)
        set_seed(self.env.args.seed)
        # Create output directory if needed
        if self.env.is_local_master():
            os.makedirs(self.env.args.output_dir, exist_ok=True)
        if self.env.is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            model.config.xla_device = True


class TaskTrainStep():
    def __init__(self, trainer_env: TrainerEnv, optimizers: TaskTrainerOptimizers, total_steps):
        self._env = trainer_env
        self._optimizers = optimizers
        self._total_steps = total_steps

    def _is_step(self, step):
        return (step + 1) % self._env.args.gradient_accumulation_steps == 0 or (
            # last step in epoch but step is always smaller than gradient_accumulation_steps
                self._total_steps <= self._env.args.gradient_accumulation_steps
                and (step + 1) == self._total_steps
        )

    def execute(self, step, inputs: Dict[str, torch.Tensor]) -> (float, bool):
        loss = self._optimizers.step_backward(inputs)

        if not self._is_step(step): return loss, False

        self._optimizers.clip_and_step()

        return loss, True


class TaskTrainEpoch():
    def __init__(self, epoch, taskContext: TaskTrainerContext):
        self._env = taskContext.env
        self._context = taskContext
        self.epoch_iterator = tqdm(taskContext.data_loaders.get_train_dataloader(), desc="Iteration",
                                   disable=not self._env.is_local_master())
        self.epoch = epoch
        self.epoch_len = len(self.epoch_iterator)
        self._train_counter = taskContext.train_counter
        self._trainer_step = self.__create_step()

    def __create_step(self):
        return TaskTrainStep(self._env, self._context.optimizers, self.epoch_len)

    def execute(self):
        for step, inputs in enumerate(self.epoch_iterator):
            if not self._train_counter.check_step_for_break(self.epoch_iterator):
                break

            if self._train_counter.check_step_for_skip():
                continue

            loss, do_step = self._trainer_step.execute(step, inputs)
            self._process_step(step, do_step, loss)

        self.epoch_iterator.close()
        self._context.trainer_logger.log_train_epoch()

    def _process_step(self, step, do_step, loss):
        self._train_counter.on_train_step(step, self.epoch_len, loss)

        if do_step and self._env.is_local_master():
            self._process_step(self.epoch, self._train_counter.tr_loss,
                               self._train_counter.logging_loss)
            self._train_counter.after_process_step()

    def _process_step_loss(self, epoch, tr_loss, logging_loss):
        if self._context.trainer_logger.is_need_log_step(self._train_counter.global_step):
            logs = {}
            logs["loss"] = (tr_loss - logging_loss) / self._env.args.logging_steps
            logs["learning_rate"] = self._context.optimizers.get_scheduler().get_last_lr()[0]

            self._context.trainer_logger.log_train_step(epoch, self._train_counter.global_step, logs)

            if self._env.args.evaluate_during_training:
                self._context.evaluator.evaluate(self._context.data_loaders.get_eval_dataloader(),
                                                 description="Training setup evaluate, step={}".format(
                                                     self._train_counter.global_step),
                                                 max_steps=self._env.args.eval_max_steps)

        self._context.trainer_checkpointer.save_check_point(self._train_counter.global_step)


class TaskTrainLoop():
    def __init__(self, taskContext: TaskTrainerContext):
        self._env = taskContext.env
        self.task_context = taskContext
        self._train_counter = taskContext.train_counter

    def execute(self):
        train_iterator = trange(
            self._train_counter.epochs_trained, int(self._train_counter.num_train_epochs), desc="Epoch",
            disable=not self._env.is_local_master()
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

    def train(self):
        self._context.trainer_logger.log_pre_train(self._context.data_loaders.get_train_dataloader(),
                                                   self._context.train_counter)

        return TaskTrainLoop(self._context).execute()

    def evaluate(self):
        return self._context.evaluator.evaluate(self._context.data_loaders.get_eval_dataloader(),
                                                description="Evaluate")


class TaskTrainerFactory:
    def __init__(self, trainer_env: TrainerEnv,
                 build_context: TaskTrainerBuildContext):
        self._env = trainer_env
        self._build_context = build_context
        self._task_context = build_context.task_context()

    def create_task_trainer(self) -> TaskTrainer:
        return TaskTrainer(TaskTrainerContext(self._env, self._build_context, True))

    def _get_finetuning_id(self):
        return None

    def create_task_predictor(self) -> TaskTrainerPredictor:
        task_model = self._build_context.task_model_factory().finetuning_model(self._get_finetuning_id())

        return TaskTrainerPredictor(self._env, task_model.model,
                                    TaskTrainerDataLoaders(self._env, self._build_context.task_dataset_factory(),
                                                           self._task_context.data_collator))
