from collections import OrderedDict

from transformers import BertTokenizer, is_tf_available
from transformers.modeling_albert import AlbertConfig

from transformers.tokenization_albert import AlbertTokenizer

from ..model_base import TaskModels, ModelType, ModelTaskType, model_func

default_model = model_func(ModelType.albert, AlbertConfig, AlbertTokenizer, 'albert')
bert_model = model_func(ModelType.albert, AlbertConfig, BertTokenizer, 'albert')

'''
Attention please, some albert models are using BertTokenizer.
'''


class Albert_Task_Models_Base(TaskModels):
    MODELS = {
        "cn": [bert_model("clue/albert_chinese_tiny"),
               bert_model("clue/albert_chinese_small"),
               default_model("onePatient/albert_chinese_small"),
               bert_model("voidful/albert_chinese_tiny"),
               bert_model("voidful/albert_chinese_small"),
               bert_model("voidful/albert_chinese_base"),
               bert_model("voidful/albert_chinese_large"),
               bert_model("voidful/albert_chinese_xlarge")
               ]}


class Albert_Task_Models(Albert_Task_Models_Base):
    from transformers.modeling_albert import AlbertForMaskedLM, AlbertModel, AlbertForQuestionAnswering, \
        AlbertForTokenClassification, AlbertPreTrainedModel, AlbertForSequenceClassification
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, AlbertModel),
        (ModelTaskType.pretrain, AlbertPreTrainedModel),
        (ModelTaskType.lm_head, AlbertForMaskedLM),
        (ModelTaskType.seq_cls, AlbertForSequenceClassification),
        (ModelTaskType.token_cls, AlbertForTokenClassification),
        (ModelTaskType.qa, AlbertForQuestionAnswering),
    ])


if is_tf_available():
    class TFAlbert_Task_Models(Albert_Task_Models_Base):
        from transformers.modeling_tf_albert import TFAlbertForMaskedLM, TFAlbertModel, TFAlbertForQuestionAnswering, \
            TFAlbertPreTrainedModel, TFAlbertForSequenceClassification, TFAlbertForMultipleChoice
        MODEL_CLASSES = OrderedDict([
            (ModelTaskType.base, TFAlbertModel),
            (ModelTaskType.pretrain, TFAlbertPreTrainedModel),
            (ModelTaskType.lm_head, TFAlbertForMaskedLM),
            (ModelTaskType.seq_cls, TFAlbertForSequenceClassification),
            (ModelTaskType.multi_choice, TFAlbertForMultipleChoice),
            (ModelTaskType.qa, TFAlbertForQuestionAnswering),
        ])
