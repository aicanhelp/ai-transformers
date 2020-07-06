import json

from .trainer_base import TrainerEnv, TaskEvalPrediction, TaskPredictionOutput
from typing import Dict, NamedTuple, Optional, List, Union
import numpy as np

from transformers.data.metrics import acc_and_f1


class TaskTrainerMetrics:
    def eval_metrics(self, p: TaskPredictionOutput): raise NotImplementedError()


class DefaultTaskTrainerMetrics(TaskTrainerMetrics):
    def __init__(self, trainer_env: TrainerEnv, for_cls=True):
        self._env = trainer_env
        self._for_cls = for_cls

    def eval_metrics(self, p: TaskPredictionOutput):
        if p.predictions is not None and p.label_ids is not None:
            metrics = self._compute_metrics(TaskEvalPrediction(predictions=p.predictions, label_ids=p.label_ids))
        else:
            metrics = {}

        if len(p.eval_losses) > 0:
            metrics["eval_loss"] = np.mean(p.eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)
        p.metrics = metrics

        output = json.dumps({**metrics})
        print(output)

        if self._env.args.tpu_metrics_debug:
            self._env.tpu_metrics()

    def _compute_metrics(self, p: TaskEvalPrediction) -> Dict:
        if self._for_cls:
            preds = np.argmax(p.predictions, axis=1)
        else:
            preds = np.squeeze(p.predictions)
        return acc_and_f1(preds, p.label_ids)
