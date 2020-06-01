from collections import OrderedDict

from transformers.modeling_electra import ElectraConfig, ElectraModel, ElectraForMaskedLM, ElectraForPreTraining, \
    ElectraDiscriminatorPredictions, ElectraForTokenClassification, ElectraGeneratorPredictions
from transformers.tokenization_electra import ElectraTokenizer, ElectraTokenizerFast
from .modelingx_electra import ElectraForSequenceClassificationX

from ..model_base import TaskModels, ModelType, ModelTaskType


class Electra_Task_Models(TaskModels):
    MODEL_TYPE = ModelType.electra
    CONFIG = ElectraConfig
    MODEL_PATHS = {
        "cn": ["hfl/chinese-electra-small-discriminator",
               "hfl/chinese-electra-small-generator",
               "hfl/chinese-electra-base-discriminator",
               "hfl/chinese-electra-base-generator"
               ]}
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, ElectraModel),
        (ModelTaskType.pretrain, ElectraForPreTraining),
        (ModelTaskType.lm_head, ElectraForMaskedLM),
        (ModelTaskType.seq_cls, ElectraForSequenceClassificationX),
        (ModelTaskType.token_cls, ElectraForTokenClassification)
    ])
    TOKENIZERS = OrderedDict([
        ('default', ElectraTokenizer),
        ('fast', ElectraTokenizerFast)
    ])
