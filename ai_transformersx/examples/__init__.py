from ..transformersx_base import *
from .tasks import *
from .. import *

TASKS = dict([
    ('news', NewsSegmentTask),
    ('sentiment', SentimentTask)
])


@configclass()
class TaskRunArguments:
    task: str = field('news', 'specified the task name: ' + str(TASKS.keys()))
    action: str = field('train', 'specified the action name: train,eval,predict')


@configclass()
class ExampleTasksArguments(TaskArguments):
    runArgs: TaskRunArguments = TaskRunArguments()


def start_example_task():
    example_args: ExampleTasksArguments = parse_tasks_args(ExampleTasksArguments)
    task_instance = TASKS.get(example_args.runArgs.task)(example_args)
    task_action = example_args.runArgs.action

    eval('task_instance.' + task_action + '()')
