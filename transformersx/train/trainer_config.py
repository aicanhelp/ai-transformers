import json
import logging
from typing import Dict, Any

import torch
from transformers import is_torch_available

from ..transformersx_base import *
from .trainer_base import TrainerEvnConfig, TrainerEnv
from .trainer_checkpoint import TrainerCheckpointConfig
from .trainer_scheduler import TrainerSchedulerConfig
from .trainer_dataloaders import TrainerDataloadersConfig
from .trainer_evaluator import TrainerEvaluatorConfig
from .trainer_logger import TrainerLoggerConfig
from .trainer_optimizers import TrainerOptimizersConfig

logger = logging.getLogger(__name__)


class TrainerTrainingConfig:
    evaluate_during_training: bool = field(True, "Run evaluation during training at each logging step.")
    output_dir: str = field('/app/models/finetuning',
                            "The output directory where the model predictions and checkpoints will be written.")
    overwrite_output_dir: bool = field(
        False, "Overwrite the content of the output directory."
               "Use this to continue training if output_dir points to a checkpoint directory.")


@configclass()
class TrainerConfig:
    env_config: TrainerEvnConfig = field(TrainerEvnConfig())
    dl_config: TrainerDataloadersConfig = field(TrainerDataloadersConfig())
    eval_config: TrainerEvaluatorConfig = field(TrainerEvaluatorConfig())
    opt_config: TrainerOptimizersConfig = field(TrainerOptimizersConfig())
    chk_config: TrainerCheckpointConfig = field(TrainerCheckpointConfig())
    log_config: TrainerLoggerConfig = field(TrainerLoggerConfig())
    sch_config: TrainerSchedulerConfig = field(TrainerSchedulerConfig())
    training_config: TrainerTrainingConfig = field(TrainerTrainingConfig())

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

    @staticmethod
    def from_env(env: TrainerEnv):
        return TrainerConfig(
            env_config=env.config,
            dl_config=env.get_config(TrainerDataloadersConfig),
            eval_config=env.get_config(TrainerEvaluatorConfig),
            opt_config=env.get_config(TrainerOptimizersConfig),
            chk_config=env.get_config(TrainerCheckpointConfig),
            log_config=env.get_config(TrainerLoggerConfig),
            sch_config=env.get_config(TrainerSchedulerConfig)
        )


@configclass()
class ConfigClass:
    pass


TaskTrainingConfig = merge_fields(ConfigClass, TrainerEvnConfig,
                                  TrainerOptimizersConfig,
                                  TrainerLoggerConfig, TrainerSchedulerConfig,
                                  TrainerDataloadersConfig,
                                  TrainerEvaluatorConfig, TrainerCheckpointConfig)
