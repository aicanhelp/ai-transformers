from transformers.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING, \
    MODEL_FOR_PRETRAINING_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING
from transformersx.configuration import DownloadConfiguration
from transformers.modeling_auto import AutoConfig, AutoModel
from transformers.tokenization_auto import AutoTokenizer
from aiharness.executors import QueueExecutor
from aiharness import harnessutils as utils
from aiharness.fileutils import join_path

log = utils.getLogger('task')

all_model_mappings = [MODEL_MAPPING,
                      MODEL_WITH_LM_HEAD_MAPPING,
                      MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
                      MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                      MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                      MODEL_FOR_PRETRAINING_MAPPING,
                      MODEL_FOR_MULTIPLE_CHOICE_MAPPING
                      ]


def model_class(config):
    for mapping in all_model_mappings:
        for config_class, model_class in mapping.items():
            if isinstance(config, config_class):
                return model_class


distil_models = [
    'adamlin/bert-distil-chinese',
]

tiny_models = [
    'clue/albert_chinese_tiny',
    'voidful/albert_chinese_tiny',
    'clue/roberta_chinese_3L312_clue_tiny',
    'clue/roberta_chinese_3L768_clue_tiny',
    'clue/roberta_chinese_clue_tiny',
    'clue/roberta_chinese_pair_tiny',
    'lonePatient/roberta_chinese_clue_tiny'
]

small_models = [
    'clue/albert_chinese_small',
    'voidful/albert_chinese_small',
    'lonePatient/albert_chinese_small',
    'hfl/chinese-electra-small-discriminator',
    'hfl/chinese-electra-small-generator'
]

base_models = [
    'bert-base-chinese',
    'clue/roberta_chinese_base',
    'hfl/chinese-roberta-wwm-ext',
    'voidful/albert_chinese_base',
    'hfl/chinese-bert-wwm-ext',
    'hfl/chinese-bert-wwm',
    'hfl/chinese-electra-base-discriminator',
    'hfl/chinese-electra-base-generator',
    'hfl/chinese-xlnet-base',
    'hfl/chinese-xlnet-mid',

]

large_models = [
    'clue/roberta_chinese_large',
    'clue/xlnet_chinese_large',
    'voidful/albert_chinese_large',
    'voidful/albert_chinese_xlarge',
    'voidful/albert_chinese_xxlarge',
    'clue/roberta_chinese_clue_large',
    'clue/roberta_chinese_pair_large',
    'hfl/chinese-roberta-wwm-ext-large'
]


class Downloader():
    def __init__(self, config: DownloadConfiguration):
        self._config = config
        model_size = config.model_size
        self.models = eval(model_size + '_models')

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
