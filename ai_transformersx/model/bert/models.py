from collections import OrderedDict

from transformers.modeling_bert import BertConfig, BertModel, BertForPreTraining, BertForMaskedLM, \
    BertForMultipleChoice, BertForNextSentencePrediction, BertForTokenClassification, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer, BertTokenizerFast, BertWordPieceTokenizer

from ai_transformersx.model.model_utils import TaskModels, ModelType, ModelTaskType


class Bert_Task_Models(TaskModels):
    MODEL_TYPE = ModelType.bert
    CONFIG = BertConfig
    MODEL_PATHS = {
        "cn": ["bert-base-chinese",
               "adamlin/bert-distil-chinese",
               "hfl/chinese-bert-wwm",
               "hfl/chinese-bert-wwm-ext"
               ]}
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, BertModel),
        (ModelTaskType.pretrain, BertForPreTraining),
        (ModelTaskType.lm_head, BertForMaskedLM),
        (ModelTaskType.seq_cls, BertForSequenceClassification),
        (ModelTaskType.multi_choice, BertForMultipleChoice),
        (ModelTaskType.token_cls, BertForTokenClassification),
        (ModelTaskType.next_seq, BertForNextSentencePrediction),
    ])
    TOKENIZERS = OrderedDict([
        ('default', BertTokenizer),
        ('fast', BertTokenizerFast),
        ('word_piece', BertWordPieceTokenizer)
    ])
