from typing import Tuple

from transformers.file_utils import cached_property
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from .trainer_utils import *
from ..transformersx_base import log, configclass, field, join_path, export

try:
    import torch_xla.core.xla_model as xm

    _has_tpu = True
except ImportError:
    _has_tpu = False

if _has_tpu:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False


@configclass
class TrainerEvnConfig():
    local_rank: int = field(-1, "For distributed training: local_rank")
    tpu_num_cores: Optional[int] = field(
        None, "TPU: Number of TPU cores (automatically passed by launcher script)")
    no_cuda: bool = field(False, "Do not use CUDA even when it is available")
    seed: int = field(42, "random seed for initialization")

    fp16: bool = field(
        False,
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    fp16_opt_level: str = field(
        "O1",
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")

    per_gpu_train_batch_size: int = field(16, "Batch size per GPU/CPU for training.")
    per_gpu_eval_batch_size: int = field(16, "Batch size per GPU/CPU for evaluation.")


class TrainerEnv:
    def __init__(self, args=None):
        self.args = args
        self.config: TrainerEvnConfig = self.get_config(TrainerEvnConfig)

    def get_config(self, config_class):
        if not self.args: return config_class()
        return export(self.args, config_class)

    def is_tpu_available(self):
        return _has_tpu

    def is_apex_available(self):
        return _has_apex

    def get_tpu_sampler(self, dataset: Dataset):
        if xm.xrt_world_size() <= 1:
            return RandomSampler(dataset)
        return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())

    def xrt_world_size(self):
        return xm.xrt_world_size()

    def is_local_master(self) -> bool:
        if self.is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.config.local_rank in [-1, 0]

    def get_tpu_dataloader(self, data_loader):
        return pl.ParallelLoader(data_loader, [self.device]).per_device_loader(self.device)

    def tpu_metrics(self):
        xm.master_print(met.metrics_report())

    def apex_model_optimizer(self, model, optimizer):
        assert self.is_apex_available(), "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
        return amp.initialize(model, optimizer, opt_level=self.config.fp16_opt_level)

    def get_tpu_eval(self, preds, label_ids):
        preds = xm.mesh_reduce("eval_preds", preds, np.concatenate)
        label_ids = xm.mesh_reduce("eval_out_label_ids", label_ids, np.concatenate)
        return preds, label_ids

    def is_master_ordinal(self):
        return xm.is_master_ordinal()

    def xm_rendezvous(self, tag):
        xm.rendezvous(tag)

    def fp16_backward(self, optimizer, loss):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    def fp16_clip(self, optimizer, max_grad_norm):
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)

    def tpu_step(self, optimizer):
        xm.optimizer_step(optimizer)

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if self.is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.config.local_rank == -1 or torch.distributed.get_rank() == 0

    def num_examples(self, dataloader: Union[DataLoader, "pl.PerDeviceLoader"]) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        if self.is_tpu_available():
            assert isinstance(dataloader, pl.PerDeviceLoader)
            return len(dataloader._loader._loader.dataset)
        else:
            return len(dataloader.dataset)

    def batch_train_size(self):
        return self.config.per_gpu_train_batch_size * max(1, self.n_gpu)

    def batch_eval_size(self):
        return self.config.per_gpu_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    def _setup_devices(self) -> Tuple["torch.device", int]:
        log.info("PyTorch: setting up devices")
        if self.config.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif self.is_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        elif self.config.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.config.local_rank)
            n_gpu = 1
        return device, n_gpu

    @property
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        return self._setup_devices[1]
