from typing import Optional

from transformers import BertConfig, BertTokenizer, TRANSFORMERS_CACHE

from ai_transformersx import *
from ai_transformersx.examples import ExampleManagement, ExampleTaskBase
from ai_transformersx.examples.main import task_manager
from ai_transformersx.model.model_base import _check_and_rename_pretrained_model

import os
from shutil import copytree, rmtree


class TestTransformersxTask:
    task_args: TaskArguments = parse_tasks_args()

    def test_parse_arg(self):
        assert self.task_args.model_args.unit_test == False

    def __do_test_task_model(self, not_use_pretrained):
        self.task_args.model_args.unit_test = True
        self.task_args.model_args.not_use_pretrained = not_use_pretrained
        taskModel: TaskModel = TaskModel(self.task_args.model_args)
        assert isinstance(taskModel.config, (BertConfig))
        assert isinstance(taskModel.tokenizer, (BertTokenizer))
        return taskModel.task_model

    def test_check_and_rename_pretrained_model(self):
        test_path = './models/pretrained/test'
        copytree('./models/pretrained/clue', join_path(test_path, 'clue'))
        _check_and_rename_pretrained_model(test_path, 'clue/albert_chinese_tiny', BertTokenizer)
        rmtree(test_path)

    def test_task_model(self):
        self.task_args.model_args.model_finetuned_dir = "models/finetuning"
        self.task_args.model_args.model_pretrained_dir = "models/pretrained"
        task_model = self.__do_test_task_model(True)
        assert task_model.model_path.startswith("models/finetuning")

        test_path = './models/pretrained/test'
        copytree('./models/pretrained', test_path)
        self.task_args.model_args.model_pretrained_dir = test_path
        task_model = self.__do_test_task_model(False)
        assert not task_model.model_path.startswith("models/finetuning")
        rmtree(test_path)


    def test_parse_task_arguments(self):
        manager = ExampleManagement()
        argument_objs = manager._build_arguments()
        args = ["news", "--action=train", "--processor-args.data-dir=/test/test"]
        task_name, arguments = ComplexArguments(argument_objs).parse(args)
        assert task_name == "news"
        assert arguments.processor_args.data_dir == "/test/test"

    def test_start_example_task(self):
        task_manager.start_example_task(["news", "--action=train", "--processor-args.data-dir=/test/test"], True)
