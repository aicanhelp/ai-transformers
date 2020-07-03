from ..task_base import *


class NewsClassificationTask(TransformerTask):
    def __init__(self, task_args: TaskArguments):
        super().__init__(task_args)

    def __create_task_context(self, task_args: TaskArguments) -> TaskContext:
        return TaskContext(
            task_name='news_cls',
            data_processor=CSVDataProcessor(
                data_dir=task_args.data_args.data_dir,
                labels=["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
            ))
