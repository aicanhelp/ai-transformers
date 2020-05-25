from collections import OrderedDict

from transformers.modeling_roberta import RobertaConfig, RobertaModel, RobertaForMaskedLM, RobertaForMultipleChoice, \
    RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification
from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

from ai_transformersx.model.model_utils import TaskModels, ModelType, ModelTaskType


class Roberta_Task_Models(TaskModels):
    MODEL_TYPE = ModelType.roberta
    CONFIG = RobertaConfig
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
        (ModelTaskType.base, RobertaModel),
        (ModelTaskType.lm_head, RobertaForMaskedLM),
        (ModelTaskType.seq_cls, RobertaForSequenceClassification),
        (ModelTaskType.token_cls, RobertaForTokenClassification),
        (ModelTaskType.qa, RobertaForQuestionAnswering),
        (ModelTaskType.multi_choice, RobertaForMultipleChoice)
    ])
    TOKENIZERS = OrderedDict([
        ('default', RobertaTokenizer),
        ('fast', RobertaTokenizerFast)
    ])
