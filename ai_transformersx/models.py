from collections import OrderedDict

from dataclasses import dataclass, fields
from transformers.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING, \
    MODEL_FOR_PRETRAINING_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from ai_transformersx.configuration import DownloadConfiguration, ModelArguments, Model_Type
from transformers.modeling_auto import AutoConfig, AutoModel
from transformers.tokenization_auto import AutoTokenizer
from ai_harness.executors import QueueExecutor
from ai_harness import harnessutils as utils
from ai_harness.fileutils import join_path

log = utils.getLogger('task')

all_model_mappings = OrderedDict([(Model_Type.base, MODEL_MAPPING),
                                  (Model_Type.lm_head, MODEL_WITH_LM_HEAD_MAPPING),
                                  (Model_Type.token_cls, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                                  (Model_Type.seq_cls, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                                  (Model_Type.qa, MODEL_FOR_QUESTION_ANSWERING_MAPPING),
                                  (Model_Type.pretrain, MODEL_FOR_PRETRAINING_MAPPING),
                                  (Model_Type.multi_choice, MODEL_FOR_MULTIPLE_CHOICE_MAPPING)
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

    def model_names(self):
        return [f.name for f in fields(self)]


@dataclass
class Model:
    path: str


@dataclass
class Distil:
    @dataclass
    class Bert(Models):
        bert_adamlin: Model = Model("adamlin/bert-distil-chinese")


@dataclass
class Tiny():
    @dataclass
    class Albert(Models):
        albert_clue: Model = Model("clue/albert_chinese_tiny")
        albert_voidful: Model = Model("voidful/albert_chinese_tiny")

    @dataclass
    class Roberta(Models):
        roberta_clue: Model = Model("clue/roberta_chinese_clue_tiny")
        roberta_clue_3l312: Model = Model("clue/roberta_chinese_3L312_clue_tiny")
        roberta_clue_3l768: Model = Model("roberta_chinese_3L768_clue_tiny")
        roberta_clue_pair: Model = Model("clue/roberta_chinese_pair_tiny")
        roberta_lonePatient: Model = Model("lonePatient/roberta_chinese_clue_tiny")


@dataclass
class Small():
    @dataclass
    class Albert(Models):
        albert_clue: Model = Model("clue/albert_chinese_small")
        albert_voidful: Model = Model("voidful/albert_chinese_small")
        albert_lonePatient: Model = Model("onePatient/albert_chinese_small")

    @dataclass
    class Electra(Models):
        electra_hfl_disc: Model = Model("hfl/chinese-electra-small-discriminator")
        electra_hfl_gen: Model = Model("hfl/chinese-electra-small-generator")


@dataclass
class Base(Models):
    @dataclass
    class Bert(Models):
        bert: Model = Model("bert-base-chinese")
        bert_hfl_wwm: Model = Model("hfl/chinese-bert-wwm")
        bert_hfl_wwm_ext: Model = Model("hfl/chinese-bert-wwm-ext")

    @dataclass
    class Albert(Models):
        albert_voidful: Model = Model("voidful/albert_chinese_base")

    @dataclass
    class Roberta(Models):
        roberta_clue: Model = Model("clue/roberta_chinese_clue_base")
        roberta_hfl_wwm: Model = Model("hfl/chinese-roberta-wwm-ext")

    @dataclass
    class Electra(Models):
        electra_hfl_disc: Model = Model("hfl/chinese-electra-base-discriminator")
        electra_hfl_gen: Model = Model("hfl/chinese-electra-base-generator")

    @dataclass
    class Xlnet(Models):
        xlnet_hfl: Model = Model("hfl/chinese-xlnet-base")


@dataclass
class Middle(Models):
    @dataclass
    class Xlnet(Models):
        xlnet_hfl: Model = Model("hfl/chinese-xlnet-mid")


@dataclass
class Large(Models):
    @dataclass
    class Albert(Models):
        albert_voidful: Model = Model("voidful/albert_chinese_large")

    @dataclass
    class Roberta(Models):
        roberta_clue: Model = Model("clue/roberta_chinese_large")
        roberta_hfl_wwm_ext: Model = Model("chinese-roberta-wwm-ext-large")
        roberta_clue_clue: Model = Model("clue/roberta_chinese_clue_large")
        roberta_clue_pair: Model = Model("clue/roberta_chinese_pair_large")

    @dataclass
    class Xlnet(Models):
        xlnet_clue: Model = Model("clue/xlnet_chinese_large")


class Model_Tools:
    @staticmethod
    def models(cls):
        return [f.default for f in fields(cls)]

    @staticmethod
    def model_names(cls):
        return [f.name for f in fields(cls)]

    @staticmethod
    def model_groups(cls):
        return [item for item in cls.__dict__.values() if type(item) == type]

    @staticmethod
    def models_by_size(model_sizes: str):
        models = []
        groups = []
        model_sizes = model_sizes.split(',')
        for m in model_sizes:
            groups.extend(Model_Tools.model_groups(eval(m.capitalize())))
        for g in groups:
            models.extend(g().models())
        return models


class Downloader():
    def __init__(self, config: DownloadConfiguration):
        self._config = config
        self.models = Model_Tools.models_by_size(config.model_size)

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
