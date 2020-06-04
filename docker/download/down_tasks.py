from ai_harness.configclasses import configclass, field
from ai_harness.configuration import Arguments
from ai_harness.executors import QueueExecutor
from ai_harness.fileutils import join_path

from ai_transformersx import query_base_models, task_model


@configclass()
class DownloadConfiguration:
    model_type: str = field('bert', 'specified the model type')
    model_path: str = field('bert', 'specified the model path')
    language: str = field('cn', 'specified the language of model')
    cache_dir: str = field('nlp_models_cache', 'specified the cache dir for models')


class Downloader():
    def __init__(self, config: DownloadConfiguration):
        self._config = config
        self.all_models = []

    def __call__(self, *args, **kwargs):
        self.__download_single()

    def __download_single(self):
        self.__download_single_language("pt")
        self.__download_single_language("tf")

    def __download_single_language(self, language):
        model = task_model(model_path=self._config.model_path,
                           model_type=self._config.model_type,
                           model_task_type='base',
                           framework=self._config.framework,
                           language=language
                           )
        self._download_model(model)

    def __download_all(self, types):
        self.__build_models(types, 'pt')
        self.__build_models(types, 'tf')
        QueueExecutor(self.all_models, worker_num=4).run(self._download_models)

    def __build_models(self, types, framework):
        models = query_base_models(types=types.split(','),
                                   framework=framework,
                                   language=self._config.language)
        if not models:
            return
        for models in models.values():
            self.all_models.extend(models[self._config.language])

    def _download_models(self, models):
        [self._download_model(model) for model in models]

    def _download_model(self, model):
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
