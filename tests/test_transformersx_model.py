from transformers import BertConfig, BertForSequenceClassification

from ai_transformersx import *


class TestTransformersModel:
    def test_task_model(self):
        model = task_model("bert-base-chinese", ModelTaskType.seq_cls, )
        assert issubclass(model.config, (BertConfig))
        assert issubclass(model.model_class, BertForSequenceClassification)
