from collections import OrderedDict

from transformers import is_tf_available
from transformers.modeling_roberta import RobertaConfig

from transformers.tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast

from ..model_base import TaskModels, ModelType, ModelTaskType, model_func

default_model = model_func(ModelType.roberta, RobertaConfig, RobertaTokenizer, 'roberta')
fast_model = model_func(ModelType.roberta, RobertaConfig, RobertaTokenizerFast, 'roberta')


class Roberta_Task_Models_Base(TaskModels):
    MODELS = {
        "cn": [default_model("clue/roberta_chinese_clue_tiny"),
               default_model("clue/roberta_chinese_3L312_clue_tiny"),
               default_model("clue/roberta_chinese_pair_tiny"),
               default_model("clue/roberta_chinese_clue_base"),
               default_model("clue/roberta_chinese_large"),
               default_model("clue/roberta_chinese_clue_large"),
               default_model("clue/roberta_chinese_pair_large"),
               default_model("roberta_chinese_3L768_clue_tiny"),
               default_model("chinese-roberta-wwm-ext-large"),
               default_model("onePatient/roberta_chinese_clue_tiny"),
               default_model("hfl/chinese-roberta-wwm-ext"),
               ]}


class Roberta_Task_Models(Roberta_Task_Models_Base):
    from transformers.modeling_roberta import RobertaModel, RobertaForMaskedLM, RobertaForMultipleChoice, \
        RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, RobertaModel),
        (ModelTaskType.lm_head, RobertaForMaskedLM),
        (ModelTaskType.seq_cls, RobertaForSequenceClassification),
        (ModelTaskType.token_cls, RobertaForTokenClassification),
        (ModelTaskType.qa, RobertaForQuestionAnswering),
        (ModelTaskType.multi_choice, RobertaForMultipleChoice)
    ])


if is_tf_available():
    class TFRoberta_Task_Models(Roberta_Task_Models_Base):
        from transformers.modeling_tf_roberta import TFRobertaModel, TFRobertaForMaskedLM, \
            TFRobertaForQuestionAnswering, TFRobertaForSequenceClassification, TFRobertaForTokenClassification
        MODEL_CLASSES = OrderedDict([
            (ModelTaskType.base, TFRobertaModel),
            (ModelTaskType.lm_head, TFRobertaForMaskedLM),
            (ModelTaskType.seq_cls, TFRobertaForSequenceClassification),
            (ModelTaskType.token_cls, TFRobertaForTokenClassification),
            (ModelTaskType.qa, TFRobertaForQuestionAnswering),
        ])
