from dataclasses import dataclass, field

from ai_transformersx.legacy.configuration import parse_args, TaskArguments


@dataclass
class TestArguments:
    test_element: str = field(default='',
                              metadata={
                                  "help": "Path to pretrained model or model identifier from huggingface.co/models"}
                              )


@dataclass
class TestArguments1:
    test_element1: str = field(default='',
                               metadata={
                                   "help": "Path to pretrained model or model identifier from huggingface.co/models"}
                               )


class Test_Arguments:
    def test_no(self):
        task_args, t1, t2 = parse_args(TestArguments, TestArguments1)

        assert not task_args.training_args.do_train
        assert not task_args.model_args.model_path
