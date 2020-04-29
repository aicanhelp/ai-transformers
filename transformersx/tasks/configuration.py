from aiharness.configuration import field, configclass


@configclass()
class TasksConfiguration:
    model: str = field('electra', 'specified the model')
    model_size: str = field('base', 'specifiy the model size')
    cache_dir: str = field('nlp_models_cache', 'specified the cache dir for models')
    task_name: str = field('download_models', 'specified the task name')
