import os
import sys

from dataclasses import field, dataclass
from ai_harness import harnessutils as aiutils
from transformers import HfArgumentParser

from ai_transformersx.model.model_args import ModelArguments
from ai_transformersx.data.data_args import DataArguments
from ai_transformersx.train.training_args import TrainingArguments

log = aiutils.getLogger('task')


@dataclass
class TaskArguments:
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments


def parse_args(*extensions) -> (TaskArguments,):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments) + extensions)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args = parser.parse_args_into_dataclasses()

    if not args[2].output_dir:
        args[2].output_dir = 'models/finetuning'
    task_args = TaskArguments(args[0], args[1], args[2])
    return task_args, tuple(args[3:])
