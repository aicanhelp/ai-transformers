import os
import random
import sys
from typing import Dict

from dataclasses import dataclass
import numpy as np
import torch
from aiharness.fileutils import join_path
from transformers import HfArgumentParser, AutoConfig, AutoTokenizer, Trainer, modeling_auto, EvalPrediction
from transformers.data.metrics import acc_and_f1

from transformersx.configuration import ModelArguments, DataArguments, log
from transformers.training_args import TrainingArguments
from transformersx import models


@dataclass
class TaskArguments:
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


def parse_args() -> TaskArguments:
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        m, d, t = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        m, d, t = parser.parse_args_into_dataclasses()

    task_args = TaskArguments(m, d, t)
    return task_args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


class TaskModel:
    def __init__(self, modelArgs: ModelArguments, model_class=None):
        self._model_args = modelArgs
        self.model_path = modelArgs.model_path
        self._model_class = model_class
        self.config, self.tokenizer, self.model = self._init()

    def _init(self):
        config = AutoConfig.from_pretrained(
            self._model_args.model_path,
            num_labels=self._model_args.num_labels
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self._model_args.model_path
        )

        if not self._model_class:
            self._model_class = models.model_class(config, self._model_args.model_type)
            if not self._model_class:
                raise ValueError(
                    "Cannot find the model class for model type {} and config {}.".format(self._model_args.model_type,
                                                                                          self._model_args.model_path)
                )

        model = self._model(config)

        return config, tokenizer, model

    def _model(self, config):
        mode_path = join_path(self._model_args.model_base_dir,
                              self._model_args.model_size,
                              self._model_args.model_path)

        model_cls = self._model_class if self._model_class else models.model_class(self.config)
        return model_cls.from_pretrained(
            mode_path,
            config=config
        )

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        if self._model_args.model_mode == "classification":
            preds = np.argmax(p.predictions, axis=1)
        elif self._model_args.model_mode == "regression":
            preds = np.squeeze(p.predictions)
        return acc_and_f1(preds, p.label_ids)


class TaskData:
    def __init__(self, dataArgs: DataArguments,
                 tokenizer, datasetClass, local_rank):
        self._data_args = dataArgs
        self._tokenizer = tokenizer
        self._datasetClass = datasetClass
        self._local_rank = local_rank
        self._train_data, self._eval_data = None, None

    def train_data(self):
        if not self._train_data:
            self._train_data = self._datasetClass(self._data_args,
                                                  tokenizer=self._tokenizer,
                                                  local_rank=self._local_rank)
        return self._train_data

    def eval_data(self):
        if not self._eval_data:
            self._eval_data = self._datasetClass(self._data_args,
                                                 tokenizer=self._tokenizer,
                                                 local_rank=self._local_rank,
                                                 evaluate=True)
        return self._eval_data


class TaskTrainer:
    def __init__(self, trainingArgs: TrainingArguments,
                 taskModel: TaskModel, taskData: TaskData, compute_metrics=None):
        self._training_args = trainingArgs
        self._taskModel = taskModel
        self._taskData = taskData
        self._compute_metrics = compute_metrics if compute_metrics else taskModel.compute_metrics
        self._trainer = self._create_trainer()

    def _check_train_args(self):
        if (
                os.path.exists(self._training_args.output_dir)
                and os.listdir(self._training_args.output_dir)
                and self._training_args.do_train
                and not self._training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({self._training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

    def _create_trainer(self):
        return Trainer(
            model=self._taskModel.model,
            args=self._training_args,
            train_dataset=self._taskData.train_data() if self._training_args.do_train else None,
            eval_dataset=self._taskData.eval_data() if self._training_args.do_eval else None,
            compute_metrics=self._compute_metrics,
        )

    def train(self):
        if self._training_args.do_train:
            self._trainer.train(
                model_path=self._taskModel.model_path
            )
            self._trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if self._trainer.is_world_master():
                self._taskModel.tokenizer.save_pretrained(self._training_args.output_dir)
        return self

    def eval(self):
        # Evaluation
        results = {}
        if self._training_args.do_eval and self._training_args.local_rank in [-1, 0]:
            log.info("*** Evaluate ***")

            eval_datasets = [self._taskData.eval_data()]

            for eval_dataset in eval_datasets:
                result = self._trainer.evaluate(eval_dataset=eval_dataset)

                output_eval_file = join_path(
                    self._training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
                )
                with open(output_eval_file, "w") as writer:
                    log.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                    for key, value in result.items():
                        log.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

                results.update(result)
        return results


class Task:
    def __init__(self, task_args: TaskArguments,
                 datasetClass, model_class=None, compute_metric=None, train=True):
        self._taskModel = TaskModel(task_args.model_args, model_class)
        self._taskData = TaskData(task_args.data_args, self._taskModel.tokenizer, datasetClass,
                                  task_args.training_args.local_rank) if task_args.training_args.do_train else None
        self._taskTrainer = TaskTrainer(task_args.training_args, self._taskModel,
                                        self._taskData, compute_metric) if task_args.training_args.do_train else None
        self._training_args = task_args.training_args

    def _log_task_start(self):
        log.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            self._training_args.local_rank,
            self._training_args.device,
            self._training_args.n_gpu,
            bool(self._training_args.local_rank != -1),
            self._training_args.fp16,
        )
        log.info("Training/evaluation parameters %s", self._training_args)

    def train(self):
        if not self._training_args.do_train:
            log.warn("do_train argument is set for False")
            return
        self._log_task_start()

        set_seed(self._training_args.seed)
        results = self._taskTrainer.train().eval()
        return results

    def predict(self, *input):
        return self._taskModel.model(*input)
