from ..task_base import *

import pandas as pd


class NewsTransformerDataProcessor(DataProcessor):
    def __init__(self, config: DataArguments):
        self._config = config

    def _get_example(self, file_name, type):
        pd_all = pd.read_csv(join_path(self._config.data_dir, file_name))

        log.info("Read data from {}, length={}".format(join_path(self._config.data_dir, file_name), len(pd_all)))
        return self._create_examples(pd_all, type)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_train_examples(self):
        return self._get_example('train.txt', 'train')

    def get_dev_examples(self):
        return self._get_example('dev.txt', 'dev')

    def get_labels(self):
        """See base class."""
        return ["体育",
                "娱乐",
                "家居",
                "房产",
                "教育",
                "时尚",
                "时政",
                "游戏",
                "科技",
                "财经"]

    def data_dir(self):
        return self._config.data_dir
