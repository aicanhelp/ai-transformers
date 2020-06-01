from ..transformersx_base import *
from .tasks import *
from .. import *

TASKS = dict([
    ('news', NewsSegmentTask),
    ('sentiment', SentimentTask)
])


@configclass
class TaskRunArguments:
    action: str = field('train', 'specified the action name: train,eval,predict')


@configclass
class ExampleTasksArguments():
    runArgs: TaskRunArguments = TaskRunArguments()
    taskArgs = None


def _build_arguments():
    arguments = {}
    for task_name, task_class in TASKS.items():
        arg_obj = ExampleTasksArguments()
        arg_obj.taskArgs = task_class.args_class()
        arguments[task_name] = arg_obj
    return arguments


def start_example_task(test=False, args=None):
    argument_objs = _build_arguments()
    args = ["-h"] if not args else args
    task_name, arguments = ComplexArguments(argument_objs).parse(args)
    arguments.taskArgs.model_args.unit_test = test
    log.info("Start Example Task:{}, with arguments:{}".format(task_name, str(arguments.taskArgs)))
    task_action = arguments.runArgs.action
    task_instance = TASKS[task_name](arguments.taskArgs)

    eval('task_instance.' + task_action + '()')
