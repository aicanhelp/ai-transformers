from ..model import TaskModelFactory
from ..train import TaskContext, TaskTrainerBuildContext, TaskTrainerFactory, TaskTrainer, \
    TaskTrainerPredictor
from ..data import TaskDatasetFactory, LocalDataStore, TaskDataConverter, DefaultTaskDataConverter
from .task_args import TaskArguments


class DefaultTaskTrainerBuildContext(TaskTrainerBuildContext):
    def __init__(self, task_args: TaskArguments, task_context: TaskContext):
        self._task_args = task_args
        self._task_context = task_context
        self._task_model_factory, self._task_data_converter, self._task_dataset_factory = None, None, None
        self._task_args.model_args.num_labels = len(self._task_context.data_processor().get_labels())

    def task_context(self) -> TaskContext:
        return self._task_context

    def task_model_factory(self) -> TaskModelFactory:
        if not self._task_model_factory:
            self._task_model_factory = TaskModelFactory(self._task_context.task_name,
                                                        self._task_args.model_args,
                                                        self._task_context.model_class())
        return self._task_model_factory

    def __get_data_store_id(self):
        return "cached_{}_{}_{}".format(
            self.task_model_factory().pretrained_model().tokenizer.__class__.__name__,
            str(self._task_args.model_args.max_len),
            self._task_context.task_name,
        )

    def task_dataset_factory(self) -> TaskDatasetFactory:
        if self._task_dataset_factory:
            return self._task_dataset_factory
        data_store_id = self.__get_data_store_id()
        task_data_store = LocalDataStore(data_store_id, self._task_args.data_args)

        self._task_dataset_factory = TaskDatasetFactory(task_data_store,
                                                        self._task_context.data_processor(),
                                                        self.task_data_converter())
        return self._task_dataset_factory

    def task_data_converter(self) -> TaskDataConverter:
        if not self._task_data_converter:
            self._task_data_converter = DefaultTaskDataConverter(self.task_model_factory().pretrained_model().tokenizer,
                                                                 self._task_context.data_processor().get_labels(),
                                                                 self._task_args.model_args.max_len,
                                                                 self._task_args.data_args.progress_bar
                                                                 )
        return self._task_data_converter


class TransformerTask:
    def __init__(self, task_args: TaskArguments, task_context: TaskContext = None):
        self._task_args = task_args
        self._task_context = self.__create_task_context(task_args) if not task_context else task_context
        self._build_context = DefaultTaskTrainerBuildContext(self._task_args, self._task_context)
        self._trainer_factory = None
        self._task_trainer, self._task_evaluator, self._task_predictor = None, None, None

    def __create_task_context(self, task_args: TaskArguments) -> TaskContext:
        return None

    def _get_trainer_factory(self):
        if self._trainer_factory:
            return self._trainer_factory
        self._trainer_factory = TaskTrainerFactory(self._task_args.training_args, self._build_context)
        return self._trainer_factory

    def _get_task_trainer(self) -> TaskTrainer:
        if not self._task_trainer:
            self._task_trainer = self._get_trainer_factory().create_task_trainer()
        return self._task_trainer

    def _get_task_predictor(self) -> TaskTrainerPredictor:
        if not self._task_predictor:
            self._task_predictor = self._get_trainer_factory().create_task_predictor()

        return self._task_predictor

    def train(self):
        self._get_task_trainer().train()

    def eval(self):
        return self._get_task_trainer().evaluate()

    def predict(self, examples):
        return self._get_task_predictor().predict(self._build_context.task_data_converter().convert(examples))
