from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel

from ..transformersx_base import log, join_path
from .models import task_model
from .model_config import TaskModelConfig


@dataclass
class TaskModel:
    config: PretrainedConfig
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel


class TaskModelFactory:
    def __init__(self, task_name, config: TaskModelConfig, model_class=None):
        self._task_name = task_name
        self.config = config
        self._model_class = model_class
        self._model_cache = {}

    def _create_task_model(self, model_class=None):
        t_model = task_model(model_path=self.config.model_name,
                             model_task_type=self.config.model_task_type,
                             language=self.config.language,
                             framework=self.config.framework)

        # use the custom class to replace the model Class
        if model_class is not None:
            t_model.model_class = model_class

        log.info("Built task model: {}".format(str(t_model)))
        return t_model

    def _load_task_model(self, model_path, model_class=None) -> TaskModel:
        t_model = self._create_task_model(model_class)
        config, tokenizer, model = t_model.load(
            num_labels=self.config.num_labels,
            unit_test=self.config.unit_test,
            cache_dir=model_path
        )
        log.info(
            "Loaded task model, config: {}, tokenizer: {}, "
            "model: {}".format(str(config),
                               type(tokenizer),
                               type(model))
        )

        return TaskModel(config=config, tokenizer=tokenizer, model=model)

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
        if model is None or not self.config.freeze_main: return

        main_parameters = eval("self.model." + model.main_parameter)
        if hasattr(main_parameters, "parameters"):
            for param in main_parameters.parameters():
                param.requires_grad = False

    def get_task_model(self, model_path=None, for_train=True) -> TaskModel:
        cached_name = 'pretrained' if not model_path else model_path
        task_model = self._model_cache.get(cached_name)

        if task_model: return task_model
        if not model_path:
            model_path = join_path(self.config.model_pretrained_dir, self.config.model_name)

        task_model = self._load_task_model(model_path, self._model_class)

        if for_train:
            self._freeze_parameters(task_model.model)
        return task_model
