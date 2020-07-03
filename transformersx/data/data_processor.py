from typing import Optional
import pandas as pd
from ..transformersx_base import join_path, log
from .data_models import TaskInputExample


class TaskDataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, limit_length: Optional[int] = None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_eval_examples(self, limit_length: Optional[int] = None):
        """Gets a collection of `InputExample`s for the eval set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class CSVDataProcessor(TaskDataProcessor):
    def __init__(self, data_dir, labels=None, label_col=0, text_a_col=1, text_b_col=-1, train_file='train.csv',
                 eval_file='dev.csv'):
        self._data_dir = data_dir
        self._labels = labels
        self._label_col = label_col
        self._text_a_col = text_a_col
        self._text_b_col = text_b_col
        self._train_file = train_file
        self._eval_file = eval_file
        self._labels = labels

    def _get_example(self, file_name, type):
        pd_all = pd.read_csv(join_path(self._data_dir, file_name))

        log.info("Read data from {}, length={}".format(join_path(self._data_dir, file_name), len(pd_all)))
        examples = []
        for i, d in enumerate(pd_all.values):
            example = TaskInputExample(guid=type + '_' + str(i),
                                       text_a=d[self._text_a_col])
            if self._label_col > -1:
                example.label = str(d[self._label_col])
            if self._text_b_col > -1:
                example.text_b = d[self._text_b_col]

            examples.append(example)

        return examples

    def get_train_examples(self, limit_length: Optional[int] = None):
        return self._get_example(self._train_file, 'train')

    def get_eval_examples(self, limit_length: Optional[int] = None):
        return self._get_example(self._eval_file, 'dev')

    def get_labels(self):
        return self._labels
