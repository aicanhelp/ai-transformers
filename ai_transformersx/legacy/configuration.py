import os
import sys
from typing import Optional

from dataclasses import field
from ai_harness import harnessutils as aiutils
from transformers import HfArgumentParser, CONFIG_MAPPING

from .training_args import TrainingArguments
from .models import *

log = aiutils.getLogger('task')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(default='',
                            metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
                            )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

    model_base_dir: str = field(default="./models/pretrained",
                                metadata={"help": "the path base dir of models"})

    model_type: str = field(default="bert",
                            metadata={
                                "help": "the type of model: " + str(CONFIG_MAPPING.keys())})

    model_task_type: str = field(default="seq_cls",
                                 metadata={
                                     "help": "the task type of model: " + str(MODEL_TASK_TYPE_NAMES)})

    model_mode: str = field(default="classification",
                            metadata={"help": "the model of model: " + str(MODEL_MODEL_NAMES)})

    model_name: str = field(default="Base.Bert.bert",
                            metadata={"help": "the name of model: " + str(ALL_MODEL_NAMES)})

    freeze_parameter: str = field(default="bert", metadata={"help": "The parameter name for freeze"})

    num_labels: str = field(default=2,
                            metadata={"help": "the number of label"})

    def validate(self):
        if not self.model_base_dir:
            raise ValueError("model_base_dir can not be empty.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(default="",
                           metadata={
                               "help": "the task name."}
                           )

    model_mode_for_data: str = field(default="classification",
                                     metadata={"help": "the model of model: classification or regression"})

    ##todo: refactor to use the max_position_embeddings of the config of model
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    predict: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    progress_bar: bool = field(default=True, metadata={"help": "Whether shows the progress_bar"})


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
