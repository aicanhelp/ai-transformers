from transformers import BertConfig, BertTokenizer

from ai_transformersx import *
from ai_transformersx.examples import ExampleManagement, ExampleTaskBase
from ai_transformersx.examples.main import task_manager


class TestTransformersxTask:
    task_args: TaskArguments = parse_tasks_args()

    def test_parse_arg(self):
        assert task_args.model_args.unit_test == False

    def __do_test_task_model(self, use_cache):
        self.task_args.model_args.unit_test = True
        self.task_args.model_args.model_base_dir = "models/finetuning"
        self.task_args.model_args.model_cache_dir = "models/pretrained"
        self.task_args.model_args.use_cache = use_cache
        taskModel: TaskModel = TaskModel(self.task_args.model_args)
        assert isinstance(taskModel.config, (BertConfig))
        assert isinstance(taskModel.tokenizer, (BertTokenizer))
        return taskModel.task_model

    def test_task_model(self):
        task_model = self.__do_test_task_model(False)
        assert task_model.model_path.startswith("models/finetuning")
        task_model = self.__do_test_task_model(True)
        assert not task_model.model_path.startswith("models/finetuning")

    def test_parse_task_arguments(self):
        manager = ExampleManagement()
        argument_objs = manager._build_arguments()
        args = ["news", "--action=train", "--processor-args.data-dir=/test/test"]
        task_name, arguments = ComplexArguments(argument_objs).parse(args)
        assert task_name == "news"
        assert arguments.processor_args.data_dir == "/test/test"

    def test_start_example_task(self):
        task_manager.start_example_task(["news", "--action=train", "--processor-args.data-dir=/test/test"], True)
