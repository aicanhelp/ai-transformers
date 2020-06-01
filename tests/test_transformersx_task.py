from transformers import BertConfig, BertTokenizer

from ai_transformersx import *
from ai_transformersx.examples import start_example_task


class TestTransformersxTask:
    task_args: TaskArguments = parse_tasks_args()

    def test_parse_arg(self):
        assert task_args.model_args.unit_test == False

    def test_task_model(self):
        self.task_args.model_args.unit_test = True
        taskModel: TaskModel = TaskModel(self.task_args.model_args)
        assert isinstance(taskModel.config, (BertConfig))
        assert isinstance(taskModel.tokenizer, (BertTokenizer))

    def test_start_example_task(self):
        start_example_task(True,["news","-h"])
