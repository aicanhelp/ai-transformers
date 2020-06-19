from typing import Optional

from transformers import BertTokenizer

from transformersx.data import *


class TestDataProcessor(TaskDataProcessor):
    def __generate_examples(self, count, type):
        return [TaskInputExample(guid=str(i), text_a=type + '_text_a_' + str(i), text_b=type + '_text_b_' + str(i),
                                 label=str(i % 2)) for i in range(count)]

    def get_train_examples(self, limit_length: Optional[int] = None):
        return self.__generate_examples(30, 'train')

    def get_eval_examples(self, limit_length: Optional[int] = None):
        return self.__generate_examples(10, 'eval')

    def get_test_examples(self, limit_length: Optional[int] = None):
        return self.__generate_examples(5, 'test')


class TestDatasetFactory:
    def __int__(self):
        data_args = TaskDataArguments()
        data_args.data_dir = '.'
        self.data_store = LocalDataStore("test", data_args)
        self.data_convertor = TaskDataConverter(BertTokenizer('../../dataset/vocab.txt'), ['0', '1'])
        self.dataset_factory = TaskDatasetFactory(self.data_store, TestDataProcessor(), self.data_convertor)

    def test_create_dataset(self):
        train_dataset = self.dataset_factory.create_train_dataset()
        eval_dataset = self.dataset_factory.create_eval_dataset()
        test_dataset = self.dataset_factory.create_predict_dataset()

    def test_load_dataset(self):
        pass
