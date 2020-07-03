from ..task_base import *


class NewsClassificationTask(TransformerTask):
    def __init__(self, config: TaskConfig):
        super().__init__(config)

    def __create_task_context(self, config: TaskConfig) -> TaskContext:
        return TaskContext(
            task_name='news_cls',
            data_processor=CSVDataProcessor(
                data_dir=config.data_config.data_dir,
                labels=["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
            ))
