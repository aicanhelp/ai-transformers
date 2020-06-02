from transformers import BertConfig, BertTokenizer

from ai_transformersx import *
from ai_transformersx.examples import ExampleManagement, ExampleTaskBase
from ai_transformersx.examples.main import manager


class TestTransformersxTask:
    task_args: TaskArguments = parse_tasks_args()

    def test_parse_arg(self):
        assert task_args.model_args.unit_test == False

    def test_task_model(self):
        self.task_args.model_args.unit_test = True
        taskModel: TaskModel = TaskModel(self.task_args.model_args)
        assert isinstance(taskModel.config, (BertConfig))
        assert isinstance(taskModel.tokenizer, (BertTokenizer))

    def test_parse_task_arguments(self):
        manager = ExampleManagement()
        argument_objs = manager._build_arguments()
        args = ["news", "--action=train", "--processor-args.data-dir=/test/test"]
        task_name, arguments = ComplexArguments(argument_objs).parse(args)
        assert task_name == "news"
        assert arguments.processor_args.data_dir == "/test/test"

    def test_start_example_task(self):
        manager.start_example_task(["news", "--action=train", "--processor-args.data-dir=/test/test"], True)
