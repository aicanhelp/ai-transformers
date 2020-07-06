from tqdm import tqdm, trange

from .trainer_base import *
from .trainer_config import TrainerConfig
from .trainer_scheduler import TaskTrainedScheduler
from .trainer_logger import TaskTrainerLogger
from .trainer_optimizers import TaskTrainerOptimizers
from .trainer_checkpoint import TaskTrainerCheckpointer
from .trainer_dataloaders import TaskTrainerDataLoaders
from .trainer_evaluator import TaskTrainerEvaluator, TaskTrainerPredictor
from ..utils import set_seed
from ..model import TaskModel
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
    def task_context(self) -> TaskContext: raise NotImplementedError()

    def task_model(self) -> TaskModel: raise NotImplementedError()

    def task_dataset_factory(self) -> TaskDatasetFactory: raise NotImplementedError()

    def task_data_converter(self) -> TaskDataConverter: raise NotImplementedError()

    def activate_context(self, model_path=None, for_train=True): raise NotImplementedError()


class TaskTrainerContextBase():
    def __init__(self, train_env: TrainerEnv, config: TrainerConfig, build_context: TaskTrainerBuildContext,
                 for_train=False):
        self.env = train_env
        self.config = config
        self.trainer_checkpointer = TaskTrainerCheckpointer(self.task_name, self.env, self.config.ckp_config)
        model_path, self.latest_checkpoint, self.task_model, self.model = self.__activate_context_and_build_model(
            build_context, for_train)
        self.task_context, self.dataset_factory = build_context.task_context(), build_context.task_dataset_factory()
        self.task_name = self.task_context.task_name

        self.data_loaders = TaskTrainerDataLoaders(self.env, self.config.dl_config, self.dataset_factory,
                                                   self.task_context.data_collator)
        self.predictor = TaskTrainerPredictor(self.env, self.model, self.data_loaders)

    def __activate_context_and_build_model(self, build_context: TaskTrainerBuildContext, for_train):
        latest_checkpoint = self.trainer_checkpointer.find_latest_checkpoint() if for_train else None
        model_path = latest_checkpoint.checkpoint_dir if latest_checkpoint else None
        model_path = None if not model_path and for_train else ""
        build_context.activate_context(model_path, for_train)
        task_model = build_context.task_model()
        self.__init_model_env(task_model.model)

        return model_path, latest_checkpoint, task_model, task_model.model

    def __init_model_env(self, model):
        model.to(self.env.device)
        set_seed(self.env.config.seed)

        if self.env.is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            model.config.xla_device = True


class TaskTrainerContext_Eval(TaskTrainerContextBase):
    def __init__(self, train_env: TrainerEnv,
                 config: TrainerConfig,
                 build_context: TaskTrainerBuildContext):
        super().__init__(train_env, config, build_context, False)
        self.evaluator = TaskTrainerEvaluator(self.env, self.config.eval_config, self.predictor,
                                              self.task_context.compute_metrics)


class TaskTrainerContext_Train(TaskTrainerContextBase):
    def __init__(self, train_env: TrainerEnv,
                 config: TrainerConfig,
                 build_context: TaskTrainerBuildContext):
        super().__init__(train_env, config, build_context, True)
        self.evaluator = TaskTrainerEvaluator(self.env, self.config.eval_config, self.predictor,
                                              self.task_context.compute_metrics)

        self.optimizers, self.train_counter = self.__create_optimizers_counter(self.model, self.latest_checkpoint)
        self.trainer_logger = TaskTrainerLogger(self.env, self.model)

    def __create_optimizers_counter(self, task_model, latest_checkpoint):
        optimizers = TaskTrainerOptimizers(self.env, task_model)
        self.config.sch_config.global_step = 0 if not latest_checkpoint else latest_checkpoint.checkpoint_number()
        self.config.sch_config.train_data_len = len(self.data_loaders.get_train_dataloader())
        self.config.sch_config.gradient_accumulation_steps = optimizers.config.gradient_accumulation_steps
        self.config.sch_config.train_batch_size = self.data_loaders.train_batch_size
        train_scheduler = TaskTrainedScheduler(self.env, self.config.sch_config)
        self.trainer_checkpointer.init_for_train(task_model, optimizers)
        self.optimizers.set_for_start_train()
        return optimizers, train_scheduler


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
    def __init__(self, epoch, taskContext: TaskTrainerContext_Train):
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
            if not self._train_counter.check_step_for_break(self.epoch_iterator): break
            if self._train_counter.check_step_for_skip(): continue

            loss, do_step = self._trainer_step.execute(step, inputs)
            self._process_step(step, do_step, loss)

        self.epoch_iterator.close()
        self._context.trainer_logger.log_train_epoch()

    def _process_step(self, step, do_step, loss):
        self._train_counter.on_train_step(step, self.epoch_len, loss)

        if do_step and self._env.is_local_master():
            self._process_step(self.epoch, self._train_counter.tr_loss, self._train_counter.logging_loss)
            self._train_counter.after_process_step()

    def _process_step_loss(self, epoch, tr_loss, logging_loss):
        if self._context.trainer_logger.is_need_log_step(self._train_counter.global_step):
            logs = {}
            logs["loss"] = (tr_loss - logging_loss) / self._env.args.logging_steps
            logs["learning_rate"] = self._context.optimizers.get_scheduler().get_last_lr()[0]

            self._context.trainer_logger.log_train_step(epoch, self._train_counter.global_step, logs)

            if self._context.config.training_config.evaluate_during_training:
                self._context.evaluator.evaluate(self._context.data_loaders.get_eval_dataloader(),
                                                 description="Training setup evaluate, step={}".format(
                                                     self._train_counter.global_step))

        self._context.trainer_checkpointer.save_check_point(self._train_counter.global_step)


class TaskTrainLoop():
    def __init__(self, taskContext: TaskTrainerContext_Train):
        self._env = taskContext.env
        self.training_config = taskContext.config.training_config
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
        self.__save_trained_model()
        return self._train_counter.task_train_output()

    def __save_trained_model(self):
        self.task_context.model.save_pretrained(self.training_config.output_dir)
        if self._env.is_world_master():
            self.task_context.task_model.tokenizer.save_pretrained(self.training_config.output_dir)


class TaskTrainer():
    def __init__(self, trainer_env: TrainerEnv, build_context: TaskTrainerBuildContext):
        self._env = trainer_env
        self.config = TrainerConfig.from_env(trainer_env)
        self._build_context = build_context

    def train(self):
        context = TaskTrainerContext_Train(self._env, self.config, self._build_context)
        context.trainer_logger.log_pre_train(context.data_loaders.get_train_dataloader(), context.train_counter)
        return TaskTrainLoop(context).execute()

    def evaluate(self):
        context = TaskTrainerContext_Eval(self._env, self.config, self._build_context)
        return context.evaluator.evaluate(context.data_loaders.get_eval_dataloader(), description="Evaluate")

    def predictor(self) -> TaskTrainerPredictor:
        predict_context = TaskTrainerContextBase(self._env, self.config, self._build_context, False)
        return predict_context.predictor
