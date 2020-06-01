import pandas as pd

from ..task_base import *


class SentimentDataProcessor(DataProcessor):
    def __init__(self, config: DataArguments):
        self._config = config

    def _get_example(self, file_name, type):
        pd_all = pd.read_csv(join_path(self._config.data_dir, file_name))

        log.info("Read data from {}, length={}".format(join_path(self._config.data_dir, file_name), len(pd_all)))
        examples = []
        for i, d in enumerate(pd_all.values):
            examples.append(InputExample(guid=type + '_' + str(i),
                                         text_a=d[1],
                                         label=str(d[0])))

        return examples

    def get_train_examples(self):
        return self._get_example('train.csv', 'train')

    def get_dev_examples(self):
        return self._get_example('dev.csv', 'dev')

    def get_labels(self):
        return ['0', '1', '2', '3']

    def data_dir(self):
        return self._config.data_dir


class SentimentTask(ExampleTaskBase):
    def __init__(self, taskArgs: TaskArguments = None):
        super().__init__(taskArgs)
        super().task_args.model_args.num_labels = 4

    def _data_processor(self):
        return SentimentDataProcessor(self.task_args.data_args)
