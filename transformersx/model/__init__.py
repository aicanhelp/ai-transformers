from collections import OrderedDict

from transformers import is_tf_available

from .electra.modelingx_electra import ElectraForSequenceClassificationX
from .bert.models import Bert_Task_Models
from .albert.models import Albert_Task_Models
from .electra.models import Electra_Task_Models
from .roberta.models import Roberta_Task_Models
from .xlnet.models import XLNet_Task_Models
from .model_base import ModelType, ModelTaskType, ModelMode, is_turbo_available

if is_tf_available():
    from .bert.models import TFBert_Task_Models
    from .albert.models import TFAlbert_Task_Models
    from .electra.models import TFElectra_Task_Models
    from .roberta.models import TFRoberta_Task_Models
    from .xlnet.models import TFXLNet_Task_Models

    TF_ALL_TASK_MODELS = OrderedDict([
        (ModelType.albert, TFAlbert_Task_Models()),
        (ModelType.bert, TFBert_Task_Models()),
        (ModelType.roberta, TFRoberta_Task_Models()),
        (ModelType.electra, TFElectra_Task_Models()),
        (ModelType.xlnet, TFXLNet_Task_Models())
    ])

ALL_TASK_MODELS = OrderedDict([
    (ModelType.albert, Albert_Task_Models()),
    (ModelType.bert, Bert_Task_Models()),
    (ModelType.roberta, Roberta_Task_Models()),
    (ModelType.electra, Electra_Task_Models()),
    (ModelType.xlnet, XLNet_Task_Models())
])

ALL_TASK_MODEL_PATHS = [task_model.all_paths() for task_model in ALL_TASK_MODELS.values()]

FRAEMEWORKS = {}

FRAEMEWORKS.setdefault("pt", ALL_TASK_MODELS)
if is_tf_available():
    FRAEMEWORKS.setdefault("tf", TF_ALL_TASK_MODELS)


def query_base_models(types, framework='pt', language='cn'):
    f_models = FRAEMEWORKS.get(framework)

    if not f_models:
        return None
    return dict([(type, f_models.get(type).all_base_models(language)) for type in types if f_models.get(type)])


def task_model(model_path, model_task_type, model_type=None, language='cn', framework='pt'):
    f_models = FRAEMEWORKS.get(framework)
    if not f_models:
        raise ValueError(
            "Cannot find the model, path={}, language={}, model_task_type={}, framework={}, Framework is not support: {}".format(
                model_path, language, model_task_type, framework, framework))
    models = f_models.get(model_type)
    if models:
        return models.task_model(model_path, model_task_type, language, framework)
    return _task_model_no_type(model_path, model_task_type, language, framework)


def _task_model_no_type(model_path, model_task_type, language='cn', framework='pt'):
    for models in FRAEMEWORKS.get(framework).values():
        model = models.task_model(model_path, model_task_type, language, ignore_not_exists=True)
        if model:
            return model
    raise ValueError("Cannot find models with model path {}".format(model_path))
