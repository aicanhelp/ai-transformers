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

    def predict(self, article: str, context_min_len=50, sentence_min_len=10):
        '''

        '''
        segment = NewsExampleSegment(article, context_min_len, sentence_min_len)
        self.task_args.data_args.predict = True
        predict_result = TransformerTask(self.task_args, PredictDataProcessor(segment),
                                         model_class=self._task_model_class
                                         ).predict()
        seperate_index = predict_result.guids * (predict_result.predictions == 0)
        return segment, seperate_index


if is_turbo_available():
    from ai_transformersx.model.bert.TurboBertSequenceClassification import TurboBertForSequenceClassification


    class TurboNewsSegmentTask(NewsSegmentTask):
        def __init__(self, taskArgs: NewsSegmentTaskArguments = None):
            super().__init__(taskArgs, TurboBertForSequenceClassification)
