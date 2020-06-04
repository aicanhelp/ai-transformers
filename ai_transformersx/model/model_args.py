from ..transformersx_base import *
from . import ALL_TASK_MODEL_PATHS, ModelTaskType, ModelMode


@configclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_base_dir: str = field("/app/models/transformers/finetuning",
                                "The 'model_base_dir' generally is for the finetuned models."
                                "Generally, the model is loaded from 'model_base_dir' firstly."
                                " If the model cannot be found, it will be loaded from the model_cache_dir.")

    model_cache_dir: str = field("/app/models/transformers/pretrained",
                                 "This folder is for the pretrained models downloaded from Internet.")

    use_cache: bool = field(True, "Use the model in model_cache_dir firstly. Generally, for training, it should be True")

    model_name: str = field("bert-base-chinese", "the name of model: " + str(ALL_TASK_MODEL_PATHS))

    model_task_type: str = field("seq_cls",
                                 "the task type of model:{}, model_task_type decides the Task".format(
                                     str(ModelTaskType.names())))

    model_mode: str = field("classification", "the model of model: " + str(ModelMode.names()))

    framework: str = field("pt", "the name of framework: pt--Pytorch,tf--Tensorflow")

    language: str = field("cn", "the language of model: cn, en")

    freeze_parameter: str = field("bert", "The parameter name for freeze")

    num_labels: int = field(2, "the number of label")

    unit_test: bool = field(False, "For unit test of the code, without the load of the model")

    def validate(self):
        if not self.model_base_dir:
            raise ValueError("model_base_dir can not be empty.")
