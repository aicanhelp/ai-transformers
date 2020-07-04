from transformers.tokenization_bert import BertTokenizer

from transformersx.data import *


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


