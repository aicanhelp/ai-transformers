from typing import Dict

from transformers.data.metrics import acc_and_f1

from transformers import EvalPrediction

from .news_data_processor import NewsDataProcessor, NewsDataConfig
import numpy as np

from ..task_base import log, TaskConfig, configclass, DefaultTransformerTask, TaskContext


@configclass
class NewsSegmentTaskArguments(TaskConfig):
    processor_config: NewsDataConfig = NewsDataConfig()


def news_compute_metrics(p: EvalPrediction) -> Dict:
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


class NewsSegmentTask(DefaultTransformerTask):
    args_class = NewsSegmentTaskArguments

    def __init__(self, config: NewsSegmentTaskArguments):
        super().__init__(config)

    def __create_task_context(self, config: NewsSegmentTaskArguments) -> TaskContext:
        return TaskContext(
            task_name='sentiment',
            data_processor=NewsDataProcessor(config.processor_config),
            compute_metrics=news_compute_metrics
        )
