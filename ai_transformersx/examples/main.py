from .tasks import *

from .examples_management import ExampleManagement

task_manager = ExampleManagement()
task_manager.register_tasks([
    ('news', NewsSegmentTask),
    ('sentiment', SentimentTask)
])

if __name__ == "__main__":
    task_manager.start_example_task()
