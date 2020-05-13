from ai_harness.configuration import Arguments
from ai_transformersx.configuration import DownloadConfiguration, log
from ai_transformersx.models import download_models


def test_task(config: DownloadConfiguration):
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


def main():
    config: DownloadConfiguration = Arguments(DownloadConfiguration()).parse()

    task_name = config.task_name
    task = get_task(task_name)
    if not task:
        log.error('Please specify the corrent task name, it should be %s, but the input is %s' % (
            str(task_names()), task_name))
        return

    task(config)


if __name__ == "__main__":
    main()
