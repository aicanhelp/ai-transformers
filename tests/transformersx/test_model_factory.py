from unittest.mock import patch
from transformersx.model.model_factory import TaskModelFactory, TaskModelConfig


class Test_Task_Model_Factory:
    @patch('transformersx.model.model_base.TaskModel')
    def test_load_model(self, mock_class):
        instance = mock_class.return_value
        instance.load.return_value = (None, None, None)
        model_factory = TaskModelFactory('test_task', TaskModelConfig())
        assert not model_factory.get_task_model().config

    @patch('transformersx.model.model_base.TaskModel')
    def test_load_model_cached(self, mock_class):
        instance = mock_class.return_value
        instance.load.return_value = ('1', None, None)
        model = TaskModelFactory('test_task', TaskModelConfig()).get_task_model()
        assert model.config == '1'
        instance.load.return_value = (None, None, None)
        assert model.config == '1'
        model = TaskModelFactory('test_task', TaskModelConfig()).get_task_model('test')
        assert not model.config
