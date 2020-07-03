from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel

from transformersx import TaskModelArguments
from ..transformersx_base import aiutils, log, join_path, configclass, field
from .models import task_model, ALL_TASK_MODEL_PATHS
from .model_base import ModelTaskType, ModelMode


@dataclass
class TaskModel:
    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel


@configclass()
class ModelConfig:
    model_name: str = field("bert-base-chinese", "the name of model: " + str(ALL_TASK_MODEL_PATHS))
    model_task_type: str = field("seq_cls",
                                 "the task type of model:{}, model_task_type decides the Task".format(
                                     str(ModelTaskType.names())))
    model_mode: str = field("classification", "the model of model: " + str(ModelMode.names()))
    framework: str = field("pt", "the name of framework: pt--Pytorch,tf--Tensorflow")
    language: str = field("cn", "the language of model: cn, en")


class BaseModelStore:
    def __init__(self, model_args: TaskModelArguments):
        self._model_args = model_args

    def _create_task_model(self, model_class=None):
        t_model = task_model(model_path=self._model_args.model_name,
                             model_task_type=self._model_args.model_task_type,
                             language=self._model_args.language,
                             framework=self._model_args.framework)

        # use the custom class to replace the model Class
        if model_class is not None:
            t_model.model_class = model_class

        log.info("Built task model: {}".format(str(t_model)))
        return t_model

    def _load_task_model(self, model_path, model_class=None) -> TaskModel:
        t_model = self._create_task_model(model_class)
        config, tokenizer, model = t_model.load(
            num_labels=self._model_args.num_labels,
            unit_test=self._model_args.unit_test,
            cache_dir=model_path
        )
        log.info(
            "Loaded task model, config: {}, tokenizer: {}, "
            "model: {}".format(str(config),
                               type(tokenizer),
                               type(model))
        )

        return TaskModel(config=config, tokenizer=tokenizer, model=model)


class PretrainedModelStore(BaseModelStore):
    def __init__(self, model_args: TaskModelArguments):
        super().__init__(model_args)
        self._store_dir = model_args.model_pretrained_dir

    def load_model(self, model_class=None, model_path=None) -> TaskModel:
        model_path = join_path(self._store_dir,
                               self._model_args.model_name) if not model_path else model_path
        task_model = self._load_task_model(model_path, model_class)
        self._freeze_parameters(task_model.model)
        return task_model

    def _freeze_parameters(self, model):
        self._freeze_weights_main(model)

        if hasattr(model, 'num_parameters'):
            log.info("num params:" + str(model.num_parameters()))
            log.info("num trainable params:" + str(model.num_parameters(only_trainable=True)))

        if hasattr(model, 'named_parameters'):
            log.info("Model Parameters Details: ")
            for name, param in model.named_parameters():
                log.info("{}:{}".format(name, param.size()))

    def _freeze_weights_main(self, model):
        if model is None or not self._model_args.freeze_main:
            return

        main_parameters = eval("self.model." + model.main_parameter)
        if hasattr(main_parameters, "parameters"):
            for param in main_parameters.parameters():
                param.requires_grad = False


class FineTuningModelStore(BaseModelStore):
    def __init__(self, model_args: TaskModelArguments):
        super().__init__(model_args)
        self._store_dir = model_args.model_finetuned_dir

    def load_model(self, task_name, fine_tuning_id=None, checkpoint_path=None, model_class=None) -> TaskModel:
        model_path = join_path(self._store_dir, task_name, self._model_args.model_name)
        if fine_tuning_id:
            model_path = join_path(model_path, fine_tuning_id)
        if checkpoint_path:
            model_path = join_path(model_path, checkpoint_path)

        return self._load_task_model(model_path, model_class)
