from ..transformersx_base import configclass, field
from .models import ALL_TASK_MODEL_PATHS
from .model_base import ModelTaskType, ModelMode


@configclass()
class TaskModelConfig:
    model_pretrained_dir: str = field("/app/models/pretrained",
                                      "This folder is for the pretrained models downloaded from Internet.")
    model_name: str = field("bert-base-chinese", "the name of model: " + str(ALL_TASK_MODEL_PATHS))
    model_task_type: str = field("seq_cls",
                                 "the task type of model:{}, model_task_type decides the Task".format(
                                     str(ModelTaskType.names())))
    model_mode: str = field("classification", "the model of model: " + str(ModelMode.names()))
    framework: str = field("pt", "the name of framework: pt--Pytorch,tf--Tensorflow")
    language: str = field("cn", "the language of model: cn, en")
    num_labels: int = field(2, "the number of label")
    max_len: int = field(512, "the max length of model input")
    unit_test: bool = field(False, "For unit test of the code, without the load of the model")
    freeze_parameters: str = field("bert", "The parameters list for freeze")
    freeze_main: bool = field(False, 'whether freeze main parameter')
