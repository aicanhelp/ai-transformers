from unittest.mock import patch
from transformersx.model.model_base import TaskModel, TaskModels
from transformersx.model.model_factory import TaskModelFactory, ModelFactoryConfig


class Test_Task_Model_Factory:
    @patch('transformersx.model.model_base.TaskModel')
    def test_load_model(self, mock_class):
        instance = mock_class.return_value
        instance.load.returnvalue = (None, None, None)
        TaskModelFactory('test_task', ModelFactoryConfig())
