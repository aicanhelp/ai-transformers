from ..transformersx_base import *


@configclass
class ServiceArgs:
    model_name = 'bert-base-chinese'
    task_name = 'task'
    port = 9091
    host = '0.0.0.0'
    models_dir = '/app/models/finetuning'
    access_log = False
    debug = False
