import numpy as np
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

from .. import *


class ExampleTaskBase:
    args_class = TaskArguments

    def __init__(self, task_name, taskArgs: TaskArguments = None, task_model_class=None):
        self._task_name = task_name
        if taskArgs is not None:
            self.task_args = taskArgs
        else:
            self.task_args = parse_tasks_args()
        self._task_model_class = task_model_class

    def __build_task(self):
        return TransformerTask(self._task_name, self.task_args, self._data_processor(),
                               model_class=self._task_model_class,
                               compute_metric=self._compute_metrics)

    def train(self):
        self.task_args.training_args.train()
        self.__build_task().train()

    def eval(self):
        self.task_args.training_args.do_eval = True
        self.__build_task().eval()

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
