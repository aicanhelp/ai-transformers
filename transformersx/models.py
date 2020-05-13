from collections import OrderedDict

from dataclasses import dataclass, fields
from transformers.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING, \
    MODEL_FOR_PRETRAINING_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from transformersx.configuration import DownloadConfiguration, ModelArguments
from transformers.modeling_auto import AutoConfig, AutoModel
from transformers.tokenization_auto import AutoTokenizer
from aiharness.executors import QueueExecutor
from aiharness import harnessutils as utils
from aiharness.fileutils import join_path

log = utils.getLogger('task')

all_model_mappings = OrderedDict([('base', MODEL_MAPPING),
                                  ('lm_head', MODEL_WITH_LM_HEAD_MAPPING),
                                  ('token_cls', MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                                  ('seq_cls', MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                                  ('qa', MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                                  ('pretrain', MODEL_FOR_PRETRAINING_MAPPING),
                                  ('multi_choice', MODEL_FOR_MULTIPLE_CHOICE_MAPPING)
                                  ])


def model_class(config, model_type):
    mapping = all_model_mappings.get(config.model_mode)
    if not mapping:
        raise ValueError(
            "Cannot find the model class for model mode {}.".format(model_type)
        )

    for config_class, model_class in mapping.items():
        if isinstance(config, config_class):
            return model_class


@dataclass
class Models:
    def models(self):
        return [f.default for f in fields(self)]


@dataclass
class Model:
    path: str


@dataclass
class Distil(Models):
    bert_adamlin: Model = Model("adamlin/bert-distil-chinese")


@dataclass
class Tiny(Models):
    albert_clue: Model = Model("clue/albert_chinese_tiny")
    albert_voidful: Model = Model("voidful/albert_chinese_tiny")
    roberta_clue: Model = Model("clue/roberta_chinese_clue_tiny")
    roberta_clue_3l312: Model = Model("clue/roberta_chinese_3L312_clue_tiny")
    roberta_clue_3l768: Model = Model("roberta_chinese_3L768_clue_tiny")
    roberta_clue_pair: Model = Model("clue/roberta_chinese_pair_tiny")
    roberta_lonePatient: Model = Model("lonePatient/roberta_chinese_clue_tiny")


@dataclass
class Small(Models):
    albert_clue: Model = Model("clue/albert_chinese_small")
    albert_voidful: Model = Model("voidful/albert_chinese_small")
    albert_lonePatient: Model = Model("onePatient/albert_chinese_small")
    electra_hfl_disc: Model = Model("hfl/chinese-electra-small-discriminator")
    electra_hfl_gen: Model = Model("hfl/chinese-electra-small-generator")


@dataclass
class Base(Models):
    bert: Model = Model("bert-base-chinese")
    bert_hfl_wwm: Model = Model("hfl/chinese-bert-wwm")
    bert_hfl_wwm_ext: Model = Model("hfl/chinese-bert-wwm-ext")
    albert_voidful: Model = Model("voidful/albert_chinese_base")
    roberta_clue: Model = Model("clue/roberta_chinese_clue_base")
    roberta_hfl_wwm: Model = Model("hfl/chinese-roberta-wwm-ext")
    electra_hfl_disc: Model = Model("hfl/chinese-electra-base-discriminator")
    electra_hfl_gen: Model = Model("hfl/chinese-electra-base-generator")
    xlnet_hfl: Model = Model("hfl/chinese-xlnet-base")


@dataclass
class Middle(Models):
    xlnet_hfl: Model = Model("hfl/chinese-xlnet-mid")


@dataclass
class Large(Models):
    albert_voidful: Model = Model("voidful/albert_chinese_large")
    roberta_clue: Model = Model("clue/roberta_chinese_large")
    roberta_hfl_wwm_ext: Model = Model("chinese-roberta-wwm-ext-large")
    roberta_clue_clue: Model = Model("clue/roberta_chinese_clue_large")
    roberta_clue_pair: Model = Model("clue/roberta_chinese_pair_large")
    xlnet_clue: Model = Model("clue/xlnet_chinese_large")


def models(model_sizes: str):
    models = []
    model_sizes = model_sizes.split(',')
    for m in model_sizes:
        models.extend(eval(m.capitalize() + '().models()'))
    return models

class Downloader():
    def __init__(self, config: DownloadConfiguration):
        self._config = config
        self.models = models(config.model_size)

    def __call__(self, *args, **kwargs):
        QueueExecutor(self.models, worker_num=8).run(self._download_model)

    def _download_model(self, model_names):
        for model_name in model_names:
            log.info('Initial model of ' + model_name)
            cache_dir = join_path(self._config.cache_dir, model_name)
            AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            AutoModel.from_pretrained(model_name, cache_dir=cache_dir)


def download_models(config: DownloadConfiguration):
    Downloader(config)()
