from collections import OrderedDict

from transformers import is_tf_available
from transformers.modeling_xlnet import XLNetConfig

from transformers.tokenization_xlnet import XLNetTokenizer

from ..model_base import TaskModels, ModelType, ModelTaskType, model_func

default_model = model_func(ModelType.electra, XLNetConfig, XLNetTokenizer, 'xlnet')


class XLNet_Task_Models_Base(TaskModels):
    MODELS = {
        "cn": [default_model("clue/roberta_chinese_clue_tiny"),
               default_model("clue/roberta_chinese_3L312_clue_tiny"),
               default_model("roberta_chinese_3L768_clue_tiny"),
               default_model("clue/roberta_chinese_pair_tiny"),
               default_model("lonePatient/roberta_chinese_clue_tiny"),
               default_model("clue/roberta_chinese_clue_base"),
               default_model("hfl/chinese-roberta-wwm-ext"),
               default_model("clue/roberta_chinese_large"),
               default_model("chinese-roberta-wwm-ext-large"),
               default_model("clue/roberta_chinese_clue_large"),
               default_model("clue/roberta_chinese_pair_large")
               ]}


class XLNet_Task_Models(XLNet_Task_Models_Base):
    from transformers.modeling_xlnet import XLNetModel, XLNetForMultipleChoice, XLNetForQuestionAnswering, \
        XLNetForQuestionAnsweringSimple, XLNetForSequenceClassification, XLNetForTokenClassification, XLNetLMHeadModel
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, XLNetModel),
        (ModelTaskType.lm_head, XLNetLMHeadModel),
        (ModelTaskType.seq_cls, XLNetForSequenceClassification),
        (ModelTaskType.token_cls, XLNetForTokenClassification),
        (ModelTaskType.qa, XLNetForQuestionAnswering),
        (ModelTaskType.qa_s, XLNetForQuestionAnsweringSimple),
        (ModelTaskType.multi_choice, XLNetForMultipleChoice)
    ])


if is_tf_available():
    class TFXLNet_Task_Models(XLNet_Task_Models_Base):
        from transformers.modeling_tf_xlnet import TFXLNetModel, TFXLNetForQuestionAnsweringSimple, \
            TFXLNetForSequenceClassification, TFXLNetForTokenClassification, TFXLNetLMHeadModel
        MODEL_CLASSES = OrderedDict([
            (ModelTaskType.base, TFXLNetModel),
            (ModelTaskType.lm_head, TFXLNetLMHeadModel),
            (ModelTaskType.seq_cls, TFXLNetForSequenceClassification),
            (ModelTaskType.token_cls, TFXLNetForTokenClassification),
            (ModelTaskType.qa_s, TFXLNetForQuestionAnsweringSimple),
        ])
