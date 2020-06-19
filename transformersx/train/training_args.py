import json
import logging
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import is_torch_available

from ..transformersx_base import *


logger = logging.getLogger(__name__)


@configclass
class TaskTrainingArguments:
    output_dir: str = field('/app/models/finetuning',
                            "The output directory where the model predictions and checkpoints will be written.")
    overwrite_output_dir: bool = field(
        False, "Overwrite the content of the output directory."
               "Use this to continue training if output_dir points to a checkpoint directory.")

    do_train: bool = field(False, "Whether to run training.")
    do_eval: bool = field(False, "Whether to run eval on the dev set.")
    do_predict: bool = field(False, "Whether to run predictions on the test set.")
    evaluate_during_training: bool = field(True, "Run evaluation during training at each logging step.")

    per_gpu_train_batch_size: int = field(16, "Batch size per GPU/CPU for training.")
    per_gpu_eval_batch_size: int = field(16, "Batch size per GPU/CPU for evaluation.")
    gradient_accumulation_steps: int = field(
        1, "Number of updates steps to accumulate before performing a backward/update pass.")

    learning_rate: float = field(5e-5, "The initial learning rate for Adam.")
    weight_decay: float = field(0.0, "Weight decay if we apply some.")
    adam_epsilon: float = field(1e-8, "Epsilon for Adam optimizer.")
    max_grad_norm: float = field(1.0, "Max gradient norm.")

    num_train_epochs: float = field(3.0, "Total number of training epochs to perform.")

    max_steps: int = field(
        -1, "If > 0: set total number of training steps to perform. Override num_train_epochs.")

    eval_max_steps: int = field(-1, "If > 0: set total number of training steps to perform.")
    warmup_steps: int = field(0, "Linear warmup over warmup_steps.")

    logging_dir: str = field(None, "Tensorboard log dir.")
    logging_first_step: bool = field(False, "Log and eval the first global_step")
    logging_steps: int = field(500, "Log every X updates steps.")
    save_steps: int = field(500, "Save checkpoint every X updates steps.")
    save_total_limit: int = field(
        None,
        "Limit the total amount of checkpoints."
        "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints")
    no_cuda: bool = field(False, "Do not use CUDA even when it is available")
    seed: int = field(42, "random seed for initialization")

    fp16: bool = field(
        False,
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    fp16_opt_level: str = field(
        "O1",
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")
    local_rank: int = field(-1, "For distributed training: local_rank")

    tpu_num_cores: Optional[int] = field(
        None, "TPU: Number of TPU cores (automatically passed by launcher script)")
    tpu_metrics_debug: bool = field(False, "TPU: Whether to print debug metrics")

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(configclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = configclasses.asdict(self)
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    def validate(self):
        if self.evaluate_during_training:
            self.do_eval = True

    def train(self):
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = True
