from collections import OrderedDict

from .electra.modelingx_electra import ElectraForSequenceClassificationX
from .bert.models import Bert_Task_Models
from .albert.models import Albert_Task_Models
from .electra.models import Electra_Task_Models
from .roberta.models import Roberta_Task_Models
from .xlnet.models import XLNet_Task_Models
from .model_utils import ModelType, ModelTaskType, ModelMode

ALL_TASK_MODELS = OrderedDict([
    (ModelType.albert, Albert_Task_Models()),
    (ModelType.bert, Bert_Task_Models()),
    (ModelType.roberta, Roberta_Task_Models()),
    (ModelType.electra, Electra_Task_Models()),
    (ModelType.xlnet, XLNet_Task_Models())
])

ALL_TASK_MODEL_PATHS = [task_model.MODEL_PATHS for task_model in ALL_TASK_MODELS.values()]


def task_model(model_type, model_path, model_task_type, tokenizer_name='default', language='cn'):
    models = ALL_TASK_MODELS.get(model_type)
    if not models:
        raise ValueError("Cannot find models with model type {}".format(model_type))
    return models.task_model(model_path, model_task_type, tokenizer_name, language)
