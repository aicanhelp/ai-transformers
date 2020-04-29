from aiharness.configuration import Arguments

from tasks.configuration import TasksConfiguration
from tasks.tasks import get_task, log, task_names


class Test_tasks:
    def test_download(self):
        config: TasksConfiguration = Arguments(TasksConfiguration()).parse()
        get_task('download_models')(config)
