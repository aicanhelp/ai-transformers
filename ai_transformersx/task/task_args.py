from ..model.model_args import ModelArguments
from ..data.data_args import DataArguments
from ..train.training_args import TrainingArguments
from ..transformersx_base import *


@configclass
class TaskArguments:
    action: str = field("train", "the task action: train,eval,predict")
    model_args: ModelArguments = ModelArguments()
    data_args: DataArguments = DataArguments()
    training_args: TrainingArguments = TrainingArguments()

def parse_tasks_args(argsObjOrClass=None):
    if argsObjOrClass is None:
        task_args = TaskArguments()
    elif argsObjOrClass is not None and type(argsObjOrClass) != type:
        task_args = argsObjOrClass
    else:
        task_args = argsObjOrClass()

    return Arguments(task_args).parse()
