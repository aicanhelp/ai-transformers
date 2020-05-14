from ai_harness.configclasses import configclass, field
from ai_harness.configuration import Arguments
from ai_harness.executors import QueueExecutor
from ai_harness.fileutils import join_path
from transformers import AutoConfig, AutoTokenizer, AutoModel

from ai_transformersx.configuration import log
from ai_transformersx.models import Model_Tools


@configclass()
class DownloadConfiguration:
    model: str = field('electra', 'specified the model')
    model_size: str = field('tiny,base', 'specifiy the model size')
    cache_dir: str = field('nlp_models_cache', 'specified the cache dir for models')
    task_name: str = field('download_models', 'specified the task name')


class Downloader():
    def __init__(self, config: DownloadConfiguration):
        self._config = config
        self.models = Model_Tools.models_by_size(config.model_size)

    def __call__(self, *args, **kwargs):
        QueueExecutor(self.models, worker_num=8).run(self._download_model)

    def _download_model(self, model_names):
        for model_name in model_names:
            log.info('Initial model of ' + model_name)
            cache_dir = join_path(self._config.cache_dir, model_name)
            AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            AutoModel.from_pretrained(model_name, cache_dir=cache_dir)


def download_models(config: DownloadConfiguration):
    Downloader(config)()


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
