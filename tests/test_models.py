from ai_transformersx.models import *


class Test_Model_Tools:
    def test_model_by_size(self):
        for model in Model_Tools.models_by_size('tiny,small'):
            print(model.path)
