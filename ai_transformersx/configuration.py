from typing import Optional

from ai_harness.configuration import field, configclass
from dataclasses import dataclass, fields
from ai_harness import harnessutils as aiutils

log = aiutils.getLogger('task')


@configclass()
class DownloadConfiguration:
    model: str = field('electra', 'specified the model')
    model_size: str = field('tiny,base', 'specifiy the model size')
    cache_dir: str = field('nlp_models_cache', 'specified the cache dir for models')
    task_name: str = field('download_models', 'specified the task name')


@dataclass
class Model_Type:
    base: str = 'base'
    pretrain: str = 'pretrain'
    lm_head: str = 'lm_head'
    qa: str = 'qa'
    seq_cls: str = 'seq_cls'
    token_cls: str = 'token_cls'
    multi_choice: str = 'multi_choice'


MODEL_TYPE_NAMES = [f.name for f in fields(Model_Type)]


@dataclass
class Model_Mode:
    classification: str = 'classification'
    regression: str = 'regression'


MODEL_MODEL_NAMES = [f.name for f in fields(Model_Mode)]


@dataclass
class Model_Size:
    distil: str = 'distil'
    tiny: str = 'tiny'
    small: str = 'small'
    base: str = 'base'
    large: str = 'large'


MODEL_SIZE_NAMES = [f.name for f in fields(Model_Size)]


@dataclass
class Model_Class:
    bert: str = 'bert'
    albert: str = 'albert'
    roberta: str = 'roberta'
    electra: str = 'electra'


MODEL_CLASS_NAMES = [f.name for f in fields(Model_Class)]


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

    model_type: str = field(default="base",
                            metadata={
                                "help": "the type of model: " + str(MODEL_TYPE_NAMES)})

    model_mode: str = field(default="classification",
                            metadata={"help": "the model of model: " + str(MODEL_MODEL_NAMES)})

    model_size: str = field(default="base",
                            metadata={"help": "the size of model: " + str(MODEL_SIZE_NAMES)})

    model_cls: str = field(default="bert",
                           metadata={"help": "the size of model: " + str(MODEL_CLASS_NAMES)})

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
