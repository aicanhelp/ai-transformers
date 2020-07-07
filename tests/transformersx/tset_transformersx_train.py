from typing import Optional
from unittest.mock import patch

from torch.utils.data import Dataset
from transformers.modeling_bert import BertConfig, BertModel, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer

from transformersx.data import TaskDatasetFactory, TaskDataConverter, DefaultTaskDataConverter, TaskDataset, \
    TaskInputExample
from transformersx.model import TaskModel
from transformersx.train import TaskTrainerBuildContext, TaskContext, TaskTrainer, TrainerEnv


def create_test_examples(type, count):
    return [TaskInputExample(guid=str(i), text_a=type + '_text_a_' + str(i), text_b=type + '_text_b_' + str(i),
                             label=str(i % 2)) for i in range(count)]


class TaskDatasetFactoryForTest(TaskDatasetFactory):
    def __init__(self, converter: TaskDataConverter):
        self._converter = converter

    def create_eval_dataset(self, limit_length: Optional[int] = None, local_rank=-1) -> Dataset:
        return TaskDataset(self._converter.convert(create_test_examples('eval', 2)))

    def create_train_dataset(self, limit_length: Optional[int] = None, local_rank=-1) -> Dataset:
        return TaskDataset(self._converter.convert(create_test_examples('train', 5)))


class TaskTrainerBuildContextForTest(TaskTrainerBuildContext):
    def __init__(self):
        self.model = self._create_task_model()
        self.converter = DefaultTaskDataConverter(self.model.tokenizer, label_list=['0', '1'], max_length=128)

    def _create_task_model(self):
        bert_config = BertConfig(vocab_size=21128, hidden_size=312, num_hidden_layers=1, num_attention_heads=4,
                                 intermediate_size=1536, max_position_embeddings=128)
        bert_tokenizer = BertTokenizer('../dataset/vocab.txt')
        bert_model = BertForSequenceClassification(bert_config)
        return TaskModel(bert_config, bert_tokenizer, bert_model)

    def task_context(self) -> TaskContext:
        return TaskContext()

    def task_dataset_factory(self) -> TaskDatasetFactory:
        return TaskDatasetFactoryForTest(self.converter)

    def task_data_converter(self) -> TaskDataConverter:
        return DefaultTaskDataConverter(self.model.tokenizer, label_list=['0', '1'])

    def task_model(self) -> TaskModel:
        return self.model

    def activate_context(self, model_path=None, for_train=True):
        pass


class Test_Task_Trainer:

    def test_predictor(self):
        assert TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest()).predictor()

    def test_predictor_with_model_path(self):
        assert TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest()).predictor('/')

    def test_evaluate(self):
        TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest()).evaluate()

    def test_evaluate_with_model_path(self):
        TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest()).evaluate('/')

    def test_train(self):
        trainer = TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest())
        trainer.config.training_config.output_dir = 'build'
        TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest()).train()

    def test_train_with_model_path(self):
        trainer = TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest())
        trainer.config.training_config.output_dir = 'build'
        TaskTrainer(TrainerEnv(), TaskTrainerBuildContextForTest()).train('/')
