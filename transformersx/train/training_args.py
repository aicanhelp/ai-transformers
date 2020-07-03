import json
import logging
from typing import Dict, Any

import torch
from transformers import is_torch_available

from ..transformersx_base import *
from .trainer_base import TrainerEvnConfig
from .trainer_checkpoint import TrainerCheckpointConfig
from .trainer_scheduler import TrainerSchedulerConfig
from .trainer_dataloaders import TrainerDataloadersConfig
from .trainer_evaluator import TrainerEvaluatorConfig
from .trainer_logger import TrainerLoggerConfig
from .trainer_optimizers import TrainerOptimizersConfig

logger = logging.getLogger(__name__)


@configclass(init=False)
class TaskTrainingArgumentsBase:
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


TaskTrainingArguments = merge_fields(TaskTrainingArgumentsBase, TrainerEvnConfig, TrainerOptimizersConfig,
                                     TrainerLoggerConfig, TrainerSchedulerConfig, TrainerDataloadersConfig,
                                     TrainerEvaluatorConfig, TrainerCheckpointConfig)
