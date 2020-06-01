import numpy as np
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

from .. import *


@configclass
class DataArguments:
    data_dir: str = field('./dataset/news', 'input the data dir for processing')
    save_mid: bool = field(True, 'whether cache the middle data for check')
    context_min_len: int = field(128, 'context min length')
    sentence_min_len: int = field(10, 'sentence min length')
    positive_mode: int = field(0, 'positive mode')
    negative_mode: int = field(0, 'negative mode')
    bar_size: int = field(1000, 'the progress bar size')


class ExampleTaskBase:
    task_args: TaskArguments = None

    def __init__(self, taskArgs: TaskArguments = None, task_model_class=None):
        if taskArgs is not None:
            self.task_args = taskArgs
        else:
            self.task_args = parse_tasks_args()
        self._task_model_class = task_model_class
        self.task_args.model_args.model_base_dir = './models/pretrained'
        self.task_args.training_args.output_dir = './models/finetuning'

    def train(self):
        self.task_args.training_args.train()

        TransformerTask(self.task_args, self._data_processor(),
                        model_class=self._task_model_class,
                        compute_metric=self._compute_metrics).train()

    def eval(self):
        self.task_args.training_args.do_eval = True
        TransformerTask(self.task_args, self._data_processor(),
                        model_class=self._task_model_class,
                        compute_metric=self._compute_metrics).eval()

    def _acc_and_f1(self, preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average="weighted")
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def _compute_metrics(self, p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        result = self._acc_and_f1(preds, p.label_ids)
        return result

    def _data_processor(self):
        raise NotImplementedError
