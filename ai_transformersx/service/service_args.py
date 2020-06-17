from ..transformersx_base import *


@configclass
class ServiceArgs:
    model_name = 'bert-base-chinese'
    task_name = ''
    concurrent_limit = 100
    port = 9091
    host = '0.0.0.0'
    context = '/'
    models_dir = '/app/models/finetuning'
    access_log = False
    debug = False
