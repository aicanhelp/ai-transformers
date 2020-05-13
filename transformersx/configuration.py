from typing import Optional

from aiharness.configuration import field, configclass
from dataclasses import dataclass
from aiharness import harnessutils as aiutils

log = aiutils.getLogger('task')


@configclass()
class DownloadConfiguration:
    model: str = field('electra', 'specified the model')
    model_size: str = field('tiny,base', 'specifiy the model size')
    cache_dir: str = field('nlp_models_cache', 'specified the cache dir for models')
    task_name: str = field('download_models', 'specified the task name')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    model_base_dir: str = field(default="",
                                metadata={"help": "the path base dir of models"})
    model_type: str = field(default=0,
                            metadata={
                                "help": "the type of model: base,pretrain,lm_head,qa,seq_cls,token_cls,multi_choice"})

    model_mode: str = field(default=0,
                            metadata={"help": "the model of model: classification or regression"})

    model_size: str = field(default="base",
                            metadata={"help": "the size of model: tiny,small,base,large"})

    num_labels: str = field(default=2,
                            metadata={"help": "the number of label"})


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
                               "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
                           )

    model_mode: str = field(default=0,
                            metadata={"help": "the model of model: classification or regression"})

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
