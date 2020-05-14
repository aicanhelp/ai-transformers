from ai_transformersx.models import *


class Test_Model_Tools:
    def test_model_by_size(self):
        for model in Model_Tools.models_by_size('tiny,small'):
            print(model.path)

    def test_model(self):
        assert Model_Tools.model('tiny', 'albert', 'albert_clue').path == 'clue/albert_chinese_tiny'

