from typing import Dict

from transformers.data.metrics import acc_and_f1

from transformers import EvalPrediction

from ai_transformersx.task import TransformerTask
from .news_data_processor import NewsDataProcessor, NewsExampleSegment, PredictDataProcessor, NewsDataArguments
import numpy as np
from ai_transformersx.model import is_turbo_available

from ..task_base import ExampleTaskBase, log, TaskArguments, configclass


@configclass
class NewsSegmentTaskArguments(TaskArguments):
    processor_args: NewsDataArguments = NewsDataArguments()


class NewsSegmentTask(ExampleTaskBase):
    args_class = NewsSegmentTaskArguments

    def __init__(self, taskArgs: NewsSegmentTaskArguments = None, task_class=None):
        super().__init__('news_segment', taskArgs, task_class)
        self._data_processor_args = taskArgs.processor_args

    def _compute_metrics(self, p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        result = acc_and_f1(preds, p.label_ids)
        log.info(p)
        log.info("preds.size=" + str(len(preds)) +
                 ", preds.sum=" + str(preds.sum()) +
                 ", label.sum=" + str(p.label_ids.sum()))

        correct = ((preds == 0) * (p.label_ids == 0)).sum()
        a1 = correct / (len(p.label_ids) - p.label_ids.sum())

        result['segment_acc'] = a1
        return result

    def _data_processor(self):
        return NewsDataProcessor(self._data_processor_args)

    def predict(self, article: str = None):
        if article:
            segment = NewsExampleSegment(article, context_min_len=self._data_processor_args.context_min_len,
                                         sentence_min_len=self._data_processor_args.sentence_min_len,
                                         check_min_anyway=self._data_processor_args.check_min_anyway)
        else:
            segment = NewsExampleSegment.from_file(join_path(self._data_processor_args.data_dir, 'dev.txt'),
                                                   context_min_len=self._data_processor_args.context_min_len,
                                                   sentence_min_len=self._data_processor_args.sentence_min_len,
                                                   check_min_anyway=self._data_processor_args.check_min_anyway,
                                                   line_sentence=False)
        self.task_args.data_args.predict = True

        p = TransformerTask('news_segment', self.task_args, PredictDataProcessor(segment)).predict()

        preds = np.argmax(p.predictions, axis=1)
        context_indexes = p.guids * (preds == 1)
        separate_indexes = [segment.contexts[i - 1][1] for i in context_indexes if i != 0]

        return segment, separate_indexes


if is_turbo_available():
    from ai_transformersx.model.bert.TurboBertSequenceClassification import TurboBertForSequenceClassification


    class TurboNewsSegmentTask(NewsSegmentTask):
        def __init__(self, taskArgs: NewsSegmentTaskArguments = None):
            super().__init__('turbo_news_segment', taskArgs, TurboBertForSequenceClassification)
