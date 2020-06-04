from dataclasses import dataclass, fields
from ai_harness import harnessutils as aiutils

log = aiutils.getLogger('task')

try:
    import turbo_transformers

    __turbo_available = True
except:
    __turbo_available = False


def is_turbo_available():
    return __turbo_available


@dataclass(frozen=True)
class ModelType:
    bert: str = "bert"
    albert: str = "albert"
    electra: str = "electra"
    roberta: str = "roberta"
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
    seq_cls_x: str = 'seq_cls_x'
    token_cls: str = 'token_cls'
    multi_choice: str = 'multi_choice'
    next_seq: str = 'next_seq'

    @staticmethod
    def names():
        return [f.name for f in fields(ModelTaskType)]


@dataclass(init=True)
class TaskModel:
    model_type: str = None
    config: type = None
    model_path: str = None
    model_class: type = None
    tokenizer: type = None

    def load(self, **kwargs):
        cache_dir = kwargs['cache_dir']
        config = self.config.from_pretrained(
            self.model_path,
            num_labels=kwargs['num_labels'],
            cache_dir=cache_dir,
            local_files_only=True
        )
        tokenizer = self.tokenizer.from_pretrained(self.model_path,
                                                   cache_dir=cache_dir,
                                                   local_files_only=True)
        unit_test = kwargs['unit_test']
        if not unit_test:
            model = self.model_class.from_pretrained(
                self.model_path,
                config=config,
                cache_dir=cache_dir,
                local_files_only=True
            )
        else:
            model = None
        return config, tokenizer, model

    def renew(self, model_class):
        return TaskModel(self.model_type, self.config, self.model_path, model_class, self.tokenizer)


def model_func(model_type, config, tokenizer):
    def generate_model(model_path):
        return TaskModel(model_type, config, model_path, None, tokenizer)

    return generate_model


class TaskModels:
    MODELS = {}
    MODEL_CLASSES = {}

    def all_base_models(self, language=None):
        return dict([(l, [model.renew(self.MODEL_CLASSES[ModelTaskType.base]) for model in models]) for l, models in
                     self.MODELS.items() if language is None or l == language])

    def all_paths(self, language=None):
        return dict([(l, [model.model_path for model in models]) for l, models in self.MODELS.items() if
                     language is None or language == l])

    def _model_class(self, model_path, model_task_type, language='cn'):
        models_l = self.MODELS[language]
        if not models_l:
            return None
        for model in models_l:
            if model.model_path == model_path:
                model_class = self.MODEL_CLASSES[model_task_type]
                if not model_class:
                    return None
                return model.renew(model_class)

        return None

    def task_model(self, model_path, model_task_type, language='cn', ignore_not_exists=False) -> TaskModel:
        model = self._model_class(model_path, model_task_type, language)
        if not model and not ignore_not_exists:
            raise ValueError(
                "Cannot find the model path {} for language {},model_type={}".format(model_path, language,
                                                                                     model_task_type))
        return model
