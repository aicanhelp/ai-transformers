import json

from ..model.model_config import TaskModelConfig
from ..data.data_config import TaskDataConfig
from ..train.trainer_config import TaskTrainingConfig
from ..transformersx_base import *


def to_json_string(obj):
    """
    Serializes this instance to a JSON string.
    """
    return json.dumps(configclasses.asdict(obj), indent=2)


@configclass
class TaskConfig:
    action: str = field("train", "the task action: train,eval,predict")
    base_dir: str = field('/app/workspace', 'task base directory')
    model_config: TaskModelConfig = TaskModelConfig()
    data_config: TaskDataConfig = TaskDataConfig()
    training_config: TaskTrainingConfig = TaskTrainingConfig()

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(configclasses.asdict(self), indent=2)


def parse_tasks_args(argsObjOrClass=None):
    if argsObjOrClass is None:
        task_args = TaskConfig()
    elif argsObjOrClass is not None and type(argsObjOrClass) != type:
        task_args = argsObjOrClass
    else:
        task_args = argsObjOrClass()

    return Arguments(task_args).parse()
