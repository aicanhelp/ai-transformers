from ..task_base import *


class SentimentTask(TransformerTask):
    def __init__(self, task_args: TaskArguments = None):
        super().__init__(task_args)

    def __create_task_context(self, task_args: TaskArguments) -> TaskContext:
        return TaskContext(
            task_name='sentiment',
            data_processor=CSVDataProcessor(
                data_dir=task_args.data_args.data_dir,
                labels=['0', '1', '2', '3']
            )
        )
