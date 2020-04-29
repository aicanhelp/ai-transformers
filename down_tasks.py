from aiharness.configuration import Arguments
from transformersx.tasks.configuration import TasksConfiguration
from transformersx.tasks.tasks import get_task, log, task_names


def main():
    config: TasksConfiguration = Arguments(TasksConfiguration()).parse()

    task_name = config.task_name
    task = get_task(task_name)
    if not task:
        log.error('Please specify the corrent task name, it should be %s, but the input is %s' % (
            str(task_names()), task_name))
        return

    task(config)


if __name__ == "__main__":
    main()
