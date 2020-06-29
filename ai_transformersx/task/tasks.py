import json
import os
import random
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers.data.metrics import simple_accuracy

from ..model import ModelMode, task_model
from ..model.model_args import ModelArguments
from ..task.task_args import TaskArguments, parse_tasks_args
from ..train.trainer import Trainer
from ..train.trainer_utils import PredictionOutput, EvalPrediction, InputExample
from ..train.training_args import TrainingArguments
from ..data import DataArguments, TaskDataset, DataProcessor
from ..transformersx_base import aiutils, log, join_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


##TODO: need to refactor the TaskModel for various cases
##(1) train a pretrained model from scratch
##(2) finetuning a task based on a pretrained Model
##(3) resume a task from the training interruption
## Currently, if need to resume the training, you must reset the model_path to the checkpoint
class TaskModel:
    def __init__(self, modelArgs: ModelArguments, model_class=None):
        self._model_args = modelArgs
        self._user_model_class = model_class
        self._init()

    def _init(self):
        self.task_model, model_cache_dir = self._build_task_model(self._model_args, self._user_model_class)
        log.info("Loading the task model from {}, {}".format(model_cache_dir, str(self.task_model)))

        self.config, self.tokenizer, self.model = self.task_model.load(
            num_labels=self._model_args.num_labels,
            unit_test=self._model_args.unit_test,
            cache_dir=model_cache_dir
        )

        log.info(
            "Loaded task model, config: {}, tokenizer: {}, "
            "model: {}".format(self.config.to_json_string(False),
                               type(self.tokenizer),
                               type(self.model))
        )

        self._freeze_parameters()

    def _freeze_parameters(self):
        # Currently, freezing the main paramerter is supported only
        self._freeze_weights_main()
        # self._freeze_parameters_others()

        if hasattr(self.model, 'num_parameters'):
            log.info("num params:" + str(self.model.num_parameters()))
            log.info("num trainable params:" + str(self.model.num_parameters(only_trainable=True)))

        if hasattr(self.model, 'named_parameters'):
            log.info("Model Parameters Details: ")
            for name, param in self.model.named_parameters():
                log.info("{}:{}".format(name, param.size()))

    def _freeze_weights_main(self):
        if self.model is None or not self._model_args.freeze_main:
            return

        main_parameters = eval("self.model." + self.task_model.main_parameter)
        if hasattr(main_parameters, "parameters"):
            for param in main_parameters.parameters():
                param.requires_grad = False

    def _freeze_weights_others(self):
        if self.model is None:
            return

        if self._model_args.freeze_parameter and hasattr(self.model, self._model_args.freeze_parameter):
            freeze_model = eval("self.model." + self._model_args.freeze_parameter)
            if hasattr(freeze_model, "parameters"):
                for param in freeze_model.parameters():
                    param.requires_grad = False

    def _build_task_model(self, modelArgs, model_class=None):
        mode_cache_dir = modelArgs.model_pretrained_dir if not modelArgs.not_use_pretrained else modelArgs.model_finetuned_dir

        if not os.path.exists(join_path(mode_cache_dir, modelArgs.model_name)):
            mode_cache_dir = modelArgs.model_pretrained_dir if modelArgs.not_use_pretrained else modelArgs.model_finetuned_dir

        if not os.path.exists(join_path(mode_cache_dir, modelArgs.model_name)):
            raise ValueError(
                "Can not find mode {} in {} or {}".format(modelArgs.model_name, modelArgs.model_finetuned_dir,
                                                          modelArgs.model_pretrained_dir)
            )

        t_model = task_model(model_path=modelArgs.model_name,
                             model_task_type=modelArgs.model_task_type,
                             language=modelArgs.language,
                             framework=modelArgs.framework)

        # use the custom class to replace the model Class
        if model_class is not None:
            t_model.model_class = model_class

        log.info("Built task model: {}".format(str(t_model)))

        return t_model, mode_cache_dir

    def _acc_and_f1(self, preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average="weighted")
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def predict(self, *input):
        return self.model(*input)

    def compute_metrics(self, p: EvalPrediction) -> Dict:
        if self._model_args.model_mode == ModelMode.classification:
            preds = np.argmax(p.predictions, axis=1)
        elif self._model_args.model_mode == ModelMode.regression:
            preds = np.squeeze(p.predictions)
        return self._acc_and_f1(preds, p.label_ids)


class TaskData:
    def __init__(self, dataArgs: DataArguments,
                 tokenizer, dataProcessor: DataProcessor, local_rank):
        self._data_args = dataArgs
        self._tokenizer = tokenizer
        self._processor = dataProcessor
        self._local_rank = local_rank
        self._train_data, self._eval_data = None, None

    def train_data(self):
        if not self._train_data and self._processor:
            self._train_data = TaskDataset(self._data_args,
                                           tokenizer=self._tokenizer,
                                           processor=self._processor,
                                           local_rank=self._local_rank)
        return self._train_data

    def eval_data(self):
        if not self._eval_data and self._processor:
            self._eval_data = TaskDataset(self._data_args,
                                          tokenizer=self._tokenizer,
                                          processor=self._processor,
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
                self._training_args.do_train
                and not self._training_args.overwrite_output_dir
                and os.path.exists(self._training_args.output_dir)
                and os.listdir(self._training_args.output_dir)
        ):
            raise ValueError(
                f"Output directory ({self._training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

    def _create_trainer(self):
        return Trainer(
            model=self._taskModel.model,
            args=self._training_args,
            train_dataset=self._taskData.train_data() if self._training_args.do_train and self._taskData else None,
            eval_dataset=self._taskData.eval_data() if self._training_args.do_eval and self._taskData else None,
            compute_metrics=self._compute_metrics,
        )

    def train(self):
        if self._training_args.do_train:
            # Here, the model_path is for the trainer to load the optimizer and scheduler states
            self._trainer.train(
                model_path=self._taskModel.task_model.model_path
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

    def predict(self, test_dataset: Dataset = None) -> PredictionOutput:
        return self._trainer.predict(test_dataset=self._taskData.eval_data() if not test_dataset else test_dataset)


class TransformerTask:
    def __init__(self, task_name, task_args: TaskArguments,
                 dataProcessor=None, model_class=None, compute_metric=None):
        log.info("Create a TransformerTask with arguments: ")
        log.info(json.dumps(task_args, default=lambda obj: obj.__dict__, indent=True))

        self.task_args = task_args
        self.task_args.validate_and_set_task_name(task_name)
        self._training_args = task_args.training_args

        self._taskModel = TaskModel(task_args.model_args, model_class)
        self._taskData = TaskData(task_args.data_args, self._taskModel.tokenizer, dataProcessor,
                                  task_args.training_args.local_rank) if dataProcessor else None
        self._taskTrainer = TaskTrainer(task_args.training_args, self._taskModel,
                                        self._taskData, compute_metric)

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

    def eval(self):
        self._log_task_start()
        set_seed(self._training_args.seed)
        results = self._taskTrainer.eval()
        return results

    def single_predict(self, example: InputExample):
        data_processor = DataProcessor()
        data_processor.get_dev_examples = lambda: [example]
        data_processor.get_labels = lambda: []
        return self.predict(data_processor)

    def predict(self, data_processor=None) -> PredictionOutput:
        dataset = TaskDataset(self.task_args.data_args,
                              tokenizer=self._taskModel.tokenizer,
                              processor=data_processor,
                              evaluate=True) if data_processor else None

        return self._taskTrainer.predict(dataset)


class DefaultTask(TransformerTask):
    def __init__(self, dataProcessorClass, model_class=None, compute_metric=None):
        super().__init__("default", parse_tasks_args(), dataProcessorClass, model_class, compute_metric)


def get_transformers_model(models_dir, model_name):
    modelArgs: ModelArguments = ModelArguments()
    modelArgs.model_name = model_name
    modelArgs.model_finetuned_dir = models_dir
    modelArgs.not_use_pretrained = True
    return TaskModel(modelArgs)
