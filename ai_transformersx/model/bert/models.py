from collections import OrderedDict

from transformers import is_tf_available
from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer, BertTokenizerFast, BertWordPieceTokenizer

from ..model_base import TaskModels, ModelType, ModelTaskType, model_func

default_model = model_func(ModelType.bert, BertConfig, BertTokenizer, 'bert')
fast_model = model_func(ModelType.bert, BertConfig, BertTokenizerFast, 'bert')
wpe_model = model_func(ModelType.bert, BertConfig, BertWordPieceTokenizer, 'bert')


class Bert_Task_Models_Base(TaskModels):
    MODELS = {
        "cn": [default_model("bert-base-chinese"),
               default_model("adamlin/bert-distil-chinese"),
               default_model("hfl/chinese-bert-wwm"),
               default_model("hfl/chinese-bert-wwm-ext")
               ]}


class Bert_Task_Models(Bert_Task_Models_Base):
    from transformers.modeling_bert import BertModel, BertForPreTraining, BertForMaskedLM, \
        BertForMultipleChoice, BertForNextSentencePrediction, BertForTokenClassification, BertForSequenceClassification
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, BertModel),
        (ModelTaskType.pretrain, BertForPreTraining),
        (ModelTaskType.lm_head, BertForMaskedLM),
        (ModelTaskType.seq_cls, BertForSequenceClassification),
        (ModelTaskType.multi_choice, BertForMultipleChoice),
        (ModelTaskType.token_cls, BertForTokenClassification),
        (ModelTaskType.next_seq, BertForNextSentencePrediction),
    ])


if is_tf_available():
    class TFBert_Task_Models(Bert_Task_Models_Base):
        from transformers.modeling_tf_bert import TFBertModel, TFBertForPreTraining, TFBertForMaskedLM, \
            TFBertForMultipleChoice, TFBertForNextSentencePrediction, TFBertForTokenClassification, \
            TFBertForSequenceClassification
        MODEL_CLASSES = OrderedDict([
            (ModelTaskType.base, TFBertModel),
            (ModelTaskType.pretrain, TFBertForPreTraining),
            (ModelTaskType.lm_head, TFBertForMaskedLM),
            (ModelTaskType.seq_cls, TFBertForSequenceClassification),
            (ModelTaskType.multi_choice, TFBertForMultipleChoice),
            (ModelTaskType.token_cls, TFBertForTokenClassification),
            (ModelTaskType.next_seq, TFBertForNextSentencePrediction),
        ])
