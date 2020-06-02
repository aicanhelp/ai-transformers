from dataclasses import dataclass, fields
from ai_harness import harnessutils as aiutils

log = aiutils.getLogger('task')


@dataclass(frozen=True)
class ModelType:
    bert: str = "bert"
    albert: str = "albert"
    electra: str = "electra"
    roberta: str = "robrerta"
    xlnet: str = "xlnet"

    @staticmethod
    def names():
        return [f.name for f in fields(ModelType)]


@dataclass
class ModelMode:
    classification: str = 'classification'
    regression: str = 'regression'

    @staticmethod
    def names():
        return [f.name for f in fields(ModelMode)]


@dataclass
class ModelTaskType:
    base: str = 'base'
    pretrain: str = 'pretrain'
    lm_head: str = 'lm_head'
    qa: str = 'qa'
    qa_s: str = 'qa_s'
    seq_cls: str = 'seq_cls'
    token_cls: str = 'token_cls'
    multi_choice: str = 'multi_choice'
    next_seq: str = 'next_seq'

    @staticmethod
    def names():
        return [f.name for f in fields(ModelTaskType)]


class TaskModel:
    def __init__(self, model_type, config, model_path, model_class, tokenizer):
        self.model_type = model_type
        self.config = config
        self.model_path = model_path
        self.model_class = model_class
        self.tokenizer = tokenizer

    def load(self, **kwargs):
        config = self.config.from_pretrained(
            self.model_path,
            num_labels=kwargs['num_labels']
        )
        tokenizer = self.tokenizer.from_pretrained(self.model_path)
        unit_test = kwargs['unit_test']
        if not unit_test:
            model = self.model_class.from_pretrained(
                self.model_path,
                config=config
            )
        else:
            model = None
        return config, tokenizer, model


class TaskModels:
    MODEL_TYPE = None
    MODEL_PATHS = {}
    MODEL_CLASSES = {}
    TOKENIZERS = {}
    CONFIG = None

    def _model_class(self, model_path, model_task_type, tokenizer_name='default', language='cn',
                     ignore_not_exists=False):
        if not model_path in self.MODEL_PATHS[language]:
            if not ignore_not_exists:
                raise ValueError(
                    "Cannot find the model path {} for language {},model_type={}".format(model_path, language,
                                                                                         self.MODEL_TYPE))
            return None

        tokenizer = self.TOKENIZERS[tokenizer_name]
        if not tokenizer:
            if not ignore_not_exists:
                raise ValueError(
                    "Cannot find the tokenizer with name {},model_type={}.".format(tokenizer_name, self.MODEL_TYPE))
            return None

        model_class = self.MODEL_CLASSES[model_task_type]
        if not model_class:
            log.warn(
                "Cannot find the model class of task type {},model_type={}.".format(model_task_type, self.MODEL_TYPE))

        return model_class, tokenizer

    def all_base_models(self, language=None):
        pass

    def task_model(self, model_path, model_task_type, tokenizer_name='default', language='cn',
                   ignore_not_exists=False) -> TaskModel:
        model_cls = self._model_class(model_path, model_task_type, tokenizer_name, language, ignore_not_exists)
        if not model_cls and ignore_not_exists:
            return None
        return TaskModel(self.MODEL_TYPE, self.CONFIG, model_path,
                         *model_cls)
