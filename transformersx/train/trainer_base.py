import json
import logging
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pip.utils import cached_property
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm.auto import tqdm, trange
from transformers import is_torch_available

from transformers.data.data_collator import DataCollator, DefaultDataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from .trainer_utils import PREFIX_CHECKPOINT_DIR, TaskEvalPrediction, TaskPredictionOutput, TaskTrainOutput, \
    TaskDefaultDataCollatorx
from .training_args import TaskTrainingArguments
from .trainer_utils import *


class TrainerBase:
    def __init__(self, args: TaskTrainingArguments):
        self._args = args

    try:
        import torch_xla.core.xla_model as xm

        _has_tpu = True
    except ImportError:
        _has_tpu = False

    def is_tpu_available(self):
        return self._has_tpu

    try:
        from apex import amp
        _has_apex = True
    except ImportError:
        _has_apex = False

    def is_apex_available(self):
        return self._has_apex

    if _has_tpu:
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import torch_xla.distributed.parallel_loader as pl

    def get_tpu_sampler(self, dataset: Dataset):
        if self.xm.xrt_world_size() <= 1:
            return RandomSampler(dataset)
        return DistributedSampler(dataset, num_replicas=self.xm.xrt_world_size(), rank=self.xm.get_ordinal())

    def is_local_master(self) -> bool:
        if self.is_tpu_available():
            return self.xm.is_master_ordinal(local=True)
        else:
            return self._args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if self.is_tpu_available():
            return self.xm.is_master_ordinal(local=False)
        else:
            return self._args.local_rank == -1 or torch.distributed.get_rank() == 0

    def num_examples(self, dataloader: Union[DataLoader, "pl.PerDeviceLoader"]) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        if self.is_tpu_available():
            assert isinstance(dataloader, self.pl.PerDeviceLoader)
            return len(dataloader._loader._loader.dataset)
        else:
            return len(dataloader.dataset)

    @property
    def train_batch_size(self) -> int:
        return self._args.per_gpu_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        return self._args.per_gpu_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self._args.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif self.is_tpu_available():
            device = self.xm.xla_device()
            n_gpu = 0
        elif self._args.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self._args.local_rank)
            n_gpu = 1
        return device, n_gpu

    @property
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        return self._setup_devices[1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for the first one (locally) to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


logger = logging.getLogger(__name__)
