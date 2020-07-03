from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel

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


class TaskModelFactory:
    def __init__(self, task_name, model_args: TaskModelArguments, model_class=None):
        self._task_name = task_name
        self._model_args = model_args
        self._pretrained_store = PretrainedModelStore(model_args)
        self._finetuning_store = FineTuningModelStore(model_args)
        self._model_class = model_class
        self._model_cache = {}

    def pretrained_model(self, model_path=None) -> TaskModel:
        task_model = self._model_cache.get('pretrained')
        if not task_model:
            task_model = self._pretrained_store.load_model(self._model_class, model_path)
            self._model_cache['pretrained'] = task_model
        return task_model

    def finetuning_model(self, fine_tuning_id=None) -> TaskModel:
        task_model = self._model_cache.get('finetuning')
        if not task_model:
            task_model = self._finetuning_store.load_model(self._task_name, fine_tuning_id, self._model_class)
            self._model_cache['finetuning'] = task_model
        return task_model
