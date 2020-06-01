import logging
from typing import Optional, Tuple

from transformers import is_torch_available
from transformers.file_utils import torch_required, cached_property

from ..transformersx_base import *

if is_torch_available():
    import torch

try:
    import torch_xla.core.xla_model as xm

    _has_tpu = True
except ImportError:
    _has_tpu = False


@torch_required
def is_tpu_available():
    return _has_tpu


logger = logging.getLogger(__name__)


@configclass
class TrainingArguments:
    output_dir: str = field('models/finetuning',
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

    @property
    def train_batch_size(self) -> int:
        return self.per_gpu_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        return self.per_gpu_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]

    def validate(self):
        if self.evaluate_during_training:
            self.do_eval = True

    def train(self):
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = True
