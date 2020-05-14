from ai_transformersx.tasks import TaskArguments, Task, parse_args
from ai_transformersx.configuration import Model_Mode, Model_Type, Model_Class, Model_Size


class Test_Task:
    def test_bert(self):
        self.task_args: TaskArguments = parse_args()
        self.task_args.model_args.model_path = './models'
        self.task_args.model_args.set_model(Model_Type.seq_cls, Model_Class.bert, Model_Size.base)
