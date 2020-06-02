from ..transformersx_base import *
from .tasks import *
from .. import *

TASKS = dict([
    ('news', NewsSegmentTask),
    ('sentiment', SentimentTask)
])


def _build_arguments():
    arguments = {}
    for task_name, task_class in TASKS.items():
        arg_obj = task_class.args_class()
        arguments[task_name] = arg_obj
    return arguments


def start_example_task(args=None, test=False):
    argument_objs = _build_arguments()

    task_name, arguments = ComplexArguments(argument_objs).parse(args)
    arguments.taskArgs.model_args.unit_test = test
    log.info("Start Example Task:{}, with arguments:{}".format(task_name, str(arguments.taskArgs)))
    task_action = arguments.runArgs.action
    task_instance = TASKS[task_name](arguments.taskArgs)

    eval('task_instance.' + task_action + '()')
