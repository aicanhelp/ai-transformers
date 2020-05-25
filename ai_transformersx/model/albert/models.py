from collections import OrderedDict

from transformers.modeling_albert import AlbertConfig, AlbertForMaskedLM, AlbertModel, AlbertForQuestionAnswering, \
    AlbertForTokenClassification, AlbertPreTrainedModel, AlbertForSequenceClassification
from transformers.tokenization_albert import AlbertTokenizer

from ai_transformersx.model.model_utils import TaskModels, ModelType, ModelTaskType


class Albert_Task_Models(TaskModels):
    MODEL_TYPE = ModelType.albert
    CONFIG = AlbertConfig
    MODEL_PATHS = {
        "cn": ["clue/albert_chinese_tiny",
               "voidful/albert_chinese_tiny",
               "clue/albert_chinese_small",
               "voidful/albert_chinese_small",
               "onePatient/albert_chinese_small",
               "voidful/albert_chinese_base",
               "voidful/albert_chinese_large"
               ]}
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, AlbertModel),
        (ModelTaskType.lm_head, AlbertForMaskedLM),
        (ModelTaskType.seq_cls, AlbertForSequenceClassification),
        (ModelTaskType.token_cls, AlbertForTokenClassification),
        (ModelTaskType.qa, AlbertForQuestionAnswering),
    ])
    TOKENIZERS = OrderedDict([
        ('default', AlbertTokenizer)
    ])
