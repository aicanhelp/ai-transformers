from transformers import InputExample

from ai_transformersx.legacy.dataprocessor import DataProcessor

from ai_transformersx.legacy.tasks import TransformerTask
from ai_transformersx.legacy.configuration import parse_args
from ai_transformersx.legacy.models import Base


class TestDataProcessor(DataProcessor):
    def get_train_examples(self):
        return [InputExample(guid='train_' + str(i), text_a="train_a_" + str(i), text_b="train_b_" + str(i),
                             label=str(i % 2)) for i in range(100)]

    def get_dev_examples(self):
        return [InputExample(guid='dev_' + str(i), text_a="dev_a_" + str(i), text_b="dev_b_" + str(i),
                             label=str(i % 2)) for i in range(10)]

    def get_labels(self):
        return ['0', '1']

    def data_dir(self):
        return ""


class Test_Task:
    def test_bert_default(self):
        task_args, _ = parse_args()
        task_args.model_args.model_base_dir = '../models/pretrained'
        TransformerTask(task_args, TestDataProcessor()).train()

    def test_bert_set1(self):
        task_args, _ = parse_args()
        task_args.model_args.model_base_dir = '../models/pretrained'
        TransformerTask(task_args, TestDataProcessor(), Base.Bert.bert).train()

    def test_bert_set2(self):
        task_args, _ = parse_args()

        task_args.model_args.model_base_dir = '../models/pretrained'
        task_args.model_args.model_name = 'Base.Bert.bert'
        TransformerTask(task_args, TestDataProcessor(), Base.Bert.bert).train()
