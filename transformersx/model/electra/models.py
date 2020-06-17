from collections import OrderedDict

from transformers import is_tf_available
from transformers.modeling_electra import ElectraConfig

from transformers.tokenization_electra import ElectraTokenizer, ElectraTokenizerFast
from .modelingx_electra import ElectraForSequenceClassificationX

from ..model_base import TaskModels, ModelType, ModelTaskType, model_func

default_model = model_func(ModelType.electra, ElectraConfig, ElectraTokenizer, 'electra')
fast_model = model_func(ModelType.electra, ElectraConfig, ElectraTokenizerFast, 'electra')


class Electra_Task_Models_Base(TaskModels):
    MODELS = {
        "cn": [default_model("hfl/chinese-electra-small-discriminator"),
               default_model("hfl/chinese-electra-small-generator"),
               default_model("hfl/chinese-electra-base-discriminator"),
               default_model("hfl/chinese-electra-base-generator"),
               default_model("hfl/chinese-electra-large-discriminator"),
               default_model("hfl/chinese-electra-large-generator")
               ]}


class Electra_Task_Models(Electra_Task_Models_Base):
    from transformers.modeling_electra import ElectraModel, ElectraForMaskedLM, ElectraForPreTraining, \
        ElectraDiscriminatorPredictions, ElectraForTokenClassification, ElectraGeneratorPredictions, \
        ElectraForSequenceClassification
    MODEL_CLASSES = OrderedDict([
        (ModelTaskType.base, ElectraModel),
        (ModelTaskType.pretrain, ElectraForPreTraining),
        (ModelTaskType.lm_head, ElectraForMaskedLM),
        (ModelTaskType.seq_cls, ElectraForSequenceClassificationX),
        (ModelTaskType.token_cls, ElectraForTokenClassification)
    ])


if is_tf_available():
    class TFElectra_Task_Models(Electra_Task_Models_Base):
        from transformers.modeling_tf_electra import TFElectraModel, TFElectraForMaskedLM, TFElectraForPreTraining, \
            TFElectraDiscriminatorPredictions, TFElectraForTokenClassification, TFElectraGeneratorPredictions
        MODEL_CLASSES = OrderedDict([
            (ModelTaskType.base, TFElectraModel),
            (ModelTaskType.pretrain, TFElectraForPreTraining),
            (ModelTaskType.lm_head, TFElectraForMaskedLM),
            (ModelTaskType.token_cls, TFElectraForTokenClassification)
        ])
