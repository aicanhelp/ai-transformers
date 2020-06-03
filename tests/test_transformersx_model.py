from transformers import BertConfig, BertForSequenceClassification

from ai_transformersx import *


class TestTransformersModel:
    def test_task_model(self):
        model = task_model("bert-base-chinese", ModelTaskType.seq_cls, )
        assert issubclass(model.config, (BertConfig))
        assert issubclass(model.model_class, BertForSequenceClassification)

    def test_models(self):
        models = query_base_models(["bert", "albert"])
        assert models["bert"] and models["albert"]
        assert models["bert"]["cn"][0].model_path.index("bert") > -1
