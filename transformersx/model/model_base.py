import fnmatch

from dataclasses import dataclass, fields
import os
from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME, url_to_filename, hf_bucket_url
from transformers.tokenization_utils import ADDED_TOKENS_FILE, SPECIAL_TOKENS_MAP_FILE, TOKENIZER_CONFIG_FILE, \
    PreTrainedTokenizer
from ..transformersx_base import log,join_path

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


def _check_and_rename_pretrained_model_file(pretrained_model_dir, model_id, file_name, use_cdn):
    target_file_path = join_path(pretrained_model_dir, model_id, file_name)
    if os.path.exists(target_file_path):
        return True
    file_url = hf_bucket_url(model_id, file_name, use_cdn=use_cdn)
    file_dir_path = join_path(pretrained_model_dir, model_id)
    url_file_name = url_to_filename(file_url)

    matching_files = [
        file
        for file in fnmatch.filter(os.listdir(file_dir_path), url_file_name + ".*")
        if not file.endswith(".json") and not file.endswith(".lock")
    ]
    if len(matching_files) > 0:
        found_file_name = join_path(file_dir_path, matching_files[-1])
        os.rename(found_file_name, target_file_path)
        return True
    return False


def _check_and_rename_pretrained_model(pretrained_model_dir, model_id, tokenizer: PreTrainedTokenizer):
    model_files = [CONFIG_NAME, WEIGHTS_NAME, ADDED_TOKENS_FILE, SPECIAL_TOKENS_MAP_FILE, TOKENIZER_CONFIG_FILE]
    model_files.extend(tokenizer.vocab_files_names.values())
    for file_name in model_files:
        if _check_and_rename_pretrained_model_file(pretrained_model_dir, model_id, file_name, False):
            continue
        _check_and_rename_pretrained_model_file(pretrained_model_dir, model_id, file_name, True)


@dataclass(init=True)
class TaskModel:
    model_type: str = None
    config: type = None
    model_path: str = None
    model_class: type = None
    tokenizer: type = None
    main_parameter: str = None

    def load(self, **kwargs):
        cache_dir = kwargs['cache_dir']
        _check_and_rename_pretrained_model(cache_dir, self.model_path, self.tokenizer)
        model_path = join_path(cache_dir, self.model_path)

        self.model_path = model_path
        config = self.config.from_pretrained(
            model_path,
            num_labels=kwargs['num_labels']
        )
        tokenizer = self.tokenizer.from_pretrained(model_path)
        unit_test = kwargs['unit_test']
        if not unit_test:
            model = self.model_class.from_pretrained(
                model_path,
                config=config
            )
        else:
            model = None
        return config, tokenizer, model

    def renew(self, model_class):
        return TaskModel(self.model_type, self.config, self.model_path,
                         model_class, self.tokenizer,
                         self.main_parameter)


def model_func(model_type, config, tokenizer, main_parameter=None):
    def generate_model(model_path):
        return TaskModel(model_type, config, model_path, None, tokenizer, main_parameter)

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
