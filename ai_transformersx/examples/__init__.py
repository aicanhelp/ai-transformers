from ..transformersx_base import *
from .tasks import *
from .. import *

TASKS = dict([
    ('news', NewsSegmentTask),
    ('sentiment', SentimentTask)
])


@configclass
class TaskRunArguments:
    func: str = field("news", "taskName")
    action: str = field('train', 'specified the action name: train,eval,predict')


@configclass
class ExampleTasksArguments():
    runArgs: TaskRunArguments = TaskRunArguments()
    taskArgs = None


def _build_arguments():
    arguments = {}
    for name, task_class in TASKS.items():
        arg_obj = ExampleTasksArguments()
        arg_obj.taskArgs = task_class.args_class()
        arguments[name] = arg_obj
    return arguments


def start_example_task(test=False, args=None):
    argument_objs = _build_arguments()
    arguments: ExampleTasksArguments = ComplexArguments(argument_objs).parse(args)
    arguments.taskArgs.model_args.unit_test = test
    log.info("Start Example Task:{}, with arguments:{}".format(arguments.runArgs.func, str(arguments.taskArgs)))
    task_action = arguments.runArgs.action
    task_instance = TASKS[arguments.runArgs.func](arguments.taskArgs)

    eval('task_instance.' + task_action + '()')
