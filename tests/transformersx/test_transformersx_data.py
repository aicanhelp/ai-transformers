from transformers.tokenization_bert import BertTokenizer

from transformersx.data import *
from transformersx.data.data_models import FeaturesSerializer
import torch
import time


def create_test_examples(type, count):
    return [TaskInputExample(guid=str(i), text_a=type + '_text_a_' + str(i), text_b=type + '_text_b_' + str(i),
                             label=str(i % 2)) for i in range(count)]


tokenizer = BertTokenizer.from_pretrained('../dataset/vocab.txt')


class Test_DefaultDataConverter:

    def test_with_bert_tokenizer(self):
        examples = create_test_examples('test', 10)
        converter = DefaultTaskDataConverter(tokenizer, label_list=['0', '1'])
        assert len(converter.convert(examples)) == 10


class Test_data_factory:
    def test_default_datafactory(self):
        config = TaskDataConfig()
        config.data_dir = 'build'
        datastore = LocalDataStore('test', config)
        data_processor = CSVDataProcessor('../dataset/label_text', labels=['0', '1'], train_file='train.txt',
                                          eval_file='dev.txt')
        converter = DefaultTaskDataConverter(tokenizer, label_list=['0', '1'])
        data_factory = TaskDatasetFactory(datastore, data_processor, converter)
        assert data_factory.create_train_dataset()
        assert data_factory.create_eval_dataset()


class Test_Serialization:
    def test_feature_serialization(self):
        features_list = [TaskInputFeatures(input_ids=[1] * (100 + i), token_type_ids=[0] * (100 + i)) for i in
                         range(5)]
        with FeaturesSerializer('../../build/test_features.data') as s:
            s.write_features_list(features_list)

        with FeaturesSerializer('../../build/test_features.data') as s:
            for i, f in enumerate(s.read_features_iter()):
                assert f == features_list[i]

    def __make_features(self, count):
        return [TaskInputFeatures(input_ids=[1] * 100, token_type_ids=[0] * 100) for i in range(count)]

    def test_benchmark_torch(self):
        features_list = self.__make_features(100000)
        time_start = time.time()
        torch.save(features_list, '../../build/test_features.data')
        print('spent time: ' + str(time.time() - time_start))

    def test_benchmark_serializer(self):
        features_list = self.__make_features(100000)
        time_start = time.time()
        with FeaturesSerializer('../../build/test_features.data') as s:
            s.write_features_list(features_list)
        print('spent time: ' + str(time.time() - time_start))
