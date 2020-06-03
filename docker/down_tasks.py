from ai_harness.configclasses import configclass, field
from ai_harness.configuration import Arguments
from ai_harness.executors import QueueExecutor
from ai_harness.fileutils import join_path

from ai_transformersx import query_base_models


@configclass()
class DownloadConfiguration:
    models: str = field('bert', 'specified the models')
    language: str = field('cn', 'specified the language of model')
    cache_dir: str = field('nlp_models_cache', 'specified the cache dir for models')


class Downloader():
    def __init__(self, config: DownloadConfiguration):
        self._config = config
        self.all_models = []

    def __call__(self, *args, **kwargs):
        self.__build_models('pt')
        self.__build_models('tf')
        QueueExecutor(self.all_models, worker_num=8).run(self._download_model)

    def __build_models(self, framework):
        models = query_base_models(types=self._config.models.split(','),
                                   framework=framework,
                                   language=self._config.language)
        if not models:
            return
        for models in models.values():
            self.all_models.extend(models[self._config.language])

    def _download_model(self, models):
        for model in models:
            cache_dir = join_path(self._config.cache_dir, model.model_path)
            self.__from_pretrained(model.config, model.model_path, cache_dir)
            self.__from_pretrained(model.tokenizer, model.model_path, cache_dir)
            self.__from_pretrained(model.model_class, model.model_path, cache_dir)

    def __from_pretrained(self, cls, path, cache_dir):
        print("Loading from {} with {}".format(path, str(cls)))
        try:
            cls.from_pretrained(path, cache_dir=cache_dir)
        except Exception as ex:
            print("Failed to load from {} with {}, error: ".format(path, str(cls), str(ex)))
            pass


if __name__ == "__main__":
    config: DownloadConfiguration = Arguments(DownloadConfiguration()).parse()

    Downloader(config)()
