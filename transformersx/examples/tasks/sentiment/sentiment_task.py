from ..task_base import *


class SentimentTask(DefaultTransformerTask):
    def __init__(self, config: TaskConfig = None):
        super().__init__(config)

    def __create_task_context(self, config: TaskConfig) -> TaskContext:
        return TaskContext(
            task_name='sentiment',
            data_processor=CSVDataProcessor(
                data_dir=config.data_config.data_dir,
                labels=['0', '1', '2', '3']
            )
        )
