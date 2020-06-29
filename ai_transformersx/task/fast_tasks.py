import torch
from fast_bert import accuracy_thresh, roc_auc, fbeta, accuracy

from .fast_task_args import FastTaskArguments
from fast_bert.learner_abs import BertAbsDataBunch, BertAbsLearner
from fast_bert.learner_cls import BertDataBunch, BertLearner
from fast_bert.learner_lm import BertLMDataBunch, BertLMLearner
from fast_bert.learner_qa import BertQADataBunch, BertQALearner
from fast_bert.prediction import BertClassificationPredictor
from ..transformersx_base import *


class TransformersFastTask():
    def __init__(self, task_name, args: FastTaskArguments):
        self._args = args
        self._task_name = task_name


class TransformersFastAbsTask(TransformersFastTask):
    def __init__(self, task_name, args: FastTaskArguments):
        super().__init__(task_name, args)


class TransformersFastClsTask(TransformersFastTask):
    def __init__(self, task_name, args: FastTaskArguments):
        super().__init__(task_name, args)
        self._model_path = join_path(self._args.model_pretrained_dir, self._args.model_name)
        self._output_dir = join_path(self._args.model_finetuned_dir, task_name, self._args.model_name)

    def metrics(self):
        metrics = []
        metrics.append({'name': 'accuracy', 'function': accuracy})
        # metrics.append({'name': 'roc_auc', 'function': roc_auc})
        # metrics.append({'name': 'fbeta', 'function': fbeta})
        return metrics

    def train(self):
        databunch = BertDataBunch(self._args.data_dir, self._args.data_dir,
                                  tokenizer=self._model_path,
                                  train_file=self._args.train_file,
                                  val_file=self._args.eval_file,
                                  label_file=self._args.labels_file,
                                  text_col=self._args.text_col,
                                  label_col=self._args.label_col,
                                  batch_size_per_gpu=self._args.batch_size_per_gpu,
                                  max_seq_length=self._args.max_seq_length,
                                  multi_gpu=True,
                                  multi_label=self._args.multi_label,
                                  model_type='bert')
        device = torch.device('cuda')
        learner = BertLearner.from_pretrained_model(databunch, self._model_path, metrics=self.metrics(),
                                                    device=device, logger=log, output_dir=self._output_dir,
                                                    finetuned_wgts_path=None, warmup_steps=5,
                                                    multi_gpu=True, is_fp16=self._args.fp16,
                                                    multi_label=self._args.multi_label, logging_steps=0)
        learner.fit(self._args.num_train_epochs, self._args.learning_rate, validate=True)
        learner.validate()


class TransformersFastLmTask(TransformersFastTask):
    def __init__(self, task_name, args: FastTaskArguments):
        super().__init__(task_name, args)


class TransformersFastQaTask(TransformersFastTask):
    def __init__(self, task_name, args: FastTaskArguments):
        super().__init__(task_name, args)
