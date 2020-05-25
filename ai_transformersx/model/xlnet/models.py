from collections import OrderedDict

from transformers.modeling_xlnet import XLNetConfig, XLNetModel, XLNetForMultipleChoice, XLNetForQuestionAnswering, \
    XLNetForQuestionAnsweringSimple, XLNetForSequenceClassification, XLNetForTokenClassification, XLNetLMHeadModel
from transformers.tokenization_xlnet import XLNetTokenizer

from ai_transformersx.model.model_utils import TaskModels, ModelType, ModelTaskType


class XLNet_Task_Models(TaskModels):
    MODEL_TYPE = ModelType.xlnet
    CONFIG = XLNetConfig
    MODEL_PATHS = {
        "cn": ["clue/roberta_chinese_clue_tiny",
               "clue/roberta_chinese_3L312_clue_tiny",
               "roberta_chinese_3L768_clue_tiny",
               "clue/roberta_chinese_pair_tiny",
               "lonePatient/roberta_chinese_clue_tiny",
               "clue/roberta_chinese_clue_base",
               "hfl/chinese-roberta-wwm-ext",
               "clue/roberta_chinese_large",
               "chinese-roberta-wwm-ext-large",
               "clue/roberta_chinese_clue_large",
               "clue/roberta_chinese_pair_large"
               ]}
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, XLNetModel),
        (ModelTaskType.lm_head, XLNetLMHeadModel),
        (ModelTaskType.seq_cls, XLNetForSequenceClassification),
        (ModelTaskType.token_cls, XLNetForTokenClassification),
        (ModelTaskType.qa, XLNetForQuestionAnswering),
        (ModelTaskType.qa_s, XLNetForQuestionAnsweringSimple),
        (ModelTaskType.multi_choice, XLNetForMultipleChoice)
    ])
    TOKENIZERS = OrderedDict([
        ('default', XLNetTokenizer)
    ])
