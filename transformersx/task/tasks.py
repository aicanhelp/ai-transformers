from ..train import TrainerEnv
from ..model import TaskModelFactory
from ..train import TaskContext, TaskTrainerBuildContext, TaskTrainer
from ..data import TaskDatasetFactory, LocalDataStore, TaskDataConverter, DefaultTaskDataConverter
from .task_config import TaskConfig


class DefaultTaskTrainerBuildContext(TaskTrainerBuildContext):
    def __init__(self, task_config: TaskConfig, task_context: TaskContext):
        self.config = task_config
        self._task_context = task_context
        self._task_model, self._task_data_converter, self._task_dataset_factory = None, None, None
        self.config.model_config.num_labels = len(self._task_context.data_processor().get_labels())

    def task_context(self) -> TaskContext:
        return self._task_context

    def task_model(self) -> TaskModelFactory:
        assert self._task_model, 'call activate_context to activate context firstly'
        return self._task_model

    def __get_data_store_id(self):
        return "cached_{}_{}_{}".format(
            self._task_model.tokenizer.__class__.__name__,
            str(self.config.model_config.max_len),
            self._task_context.task_name,
        )

    def task_dataset_factory(self) -> TaskDatasetFactory:
        if self._task_dataset_factory: return self._task_dataset_factory
        assert self._task_model, 'call activate_context to activate context firstly'
        data_store_id = self.__get_data_store_id()
        task_data_store = LocalDataStore(data_store_id, self.config.data_config)

        self._task_dataset_factory = TaskDatasetFactory(task_data_store,
                                                        self._task_context.data_processor(),
                                                        self.task_data_converter())
        return self._task_dataset_factory

    def task_data_converter(self) -> TaskDataConverter:
        if self._task_data_converter: return self._task_data_converter
        assert self._task_model, 'call activate_context to activate context firstly'
        self._task_data_converter = DefaultTaskDataConverter(
            self._task_model.tokenizer,
            self._task_context.data_processor().get_labels(),
            self.config.model_config.max_len,
            self.config.data_config.progress_bar
        )
        return self._task_data_converter

    def activate_context(self, model_path=None, for_train=True):
        if self._task_model: return self._task_model
        self._task_model = TaskModelFactory(self._task_context.task_name,
                                            self.config.model_config,
                                            self._task_context.model_class()).get_task_model(model_path, for_train)
        return self._task_model


class TransformerTask:
    args_class = TaskConfig

    def train(self): raise NotImplementedError()

    def eval(self): raise NotImplementedError()

    def predict(self, examples): raise NotImplementedError()


class DefaultTransformerTask(TransformerTask):
    args_class = TaskConfig

    def __init__(self, config: TaskConfig, task_context: TaskContext = None):
        self.config = config
        self._task_context = self.__create_task_context(config) if not task_context else task_context
        self._build_context = DefaultTaskTrainerBuildContext(self.config, self._task_context)
        self._task_trainer, self._task_predictor = None, None

    def __create_task_context(self, config: TaskConfig) -> TaskContext:
        raise NotImplementedError('this method must be implemented for None task_context argument in Construction')

    def _get_task_trainer(self) -> TaskTrainer:
        if not self._task_trainer: self._task_trainer = TaskTrainer(TrainerEnv(self.config.training_config),
                                                                    self._build_context)
        return self._task_trainer

    def train(self):
        self._get_task_trainer().train()

    def eval(self):
        return self._get_task_trainer().evaluate()

    def predict(self, examples):
        if not self._task_predictor:
            self._task_predictor = self._get_task_trainer().predictor()
        return self._task_predictor.predict(self._build_context.task_data_converter().convert(examples))
