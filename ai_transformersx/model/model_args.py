from typing import Optional

from dataclasses import field, dataclass
from ai_harness import harnessutils as aiutils

from ai_transformersx.model import *

log = aiutils.getLogger('task')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    model_base_dir: str = field(default="./models/pretrained",
                                metadata={"help": "the path base dir of models"})

    model_type: str = field(default="bert",
                            metadata={
                                "help": "the type of model: " + str(ModelType.names())})

    model_task_type: str = field(default="seq_cls",
                                 metadata={
                                     "help": "the task type of model: " + str()})

    model_mode: str = field(default="classification",
                            metadata={"help": "the model of model: " + str(ModelMode.names())})

    model_name: str = field(default="bert-base-chinese",
                            metadata={"help": "the name of model: " + str(ALL_TASK_MODEL_PATHS)})

    tokenizer_type: str = field(default="default",
                                metadata={"help": "the name of tokenizer: default,fast"})

    language: str = field(default="cn",
                          metadata={"help": "the language of model: cn, en"})

    freeze_parameter: str = field(default="bert", metadata={"help": "The parameter name for freeze"})

    num_labels: str = field(default=2,
                            metadata={"help": "the number of label"})

    def validate(self):
        if not self.model_base_dir:
            raise ValueError("model_base_dir can not be empty.")
