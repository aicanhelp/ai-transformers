from aiharness import harnessutils as aiutils
from transformersx.tasks.configuration import TasksConfiguration
from transformersx.tasks.models import download_models

log = aiutils.getLogger('task')


def test_task(config: TasksConfiguration):
    '''
        a task for test
    '''
    log.info('Run the test task with configuration: ' + str(config))


TASKS = {
    'test': test_task,
    'download_models': download_models
}


def task_names():
    return TASKS.keys()


def get_task(task_name):
    return TASKS.get(task_name)
