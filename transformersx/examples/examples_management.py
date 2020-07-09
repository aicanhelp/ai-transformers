from ai_harness.configuration import ComplexArguments
from ..task import TransformerTask
from ..transformersx_base import log, to_json_string


class ExampleManagement():
    TASKS = dict()

    def register_task(self, task_name, task_class):
        if self.TASKS.get(task_name) is not None:
            raise ValueError(
                "task name {} already exists, please use other name.".format(task_name)
            )
        self.TASKS.setdefault(task_name, task_class)
        if not issubclass(task_class, (TransformerTask)):
            raise ValueError(
                "task_class must be the subclass of TransformerTask"
            )
        return self

    def register_tasks(self, tasks):
        if not tasks:
            return
        for task in tasks:
            self.register_task(task[0], task[1])
        return self

    def _build_arguments(self):
        arguments = {}
        for task_name, task_class in self.TASKS.items():
            arg_obj = task_class.args_class()
            arguments[task_name] = arg_obj
        return arguments

    def start_example_task(self, args=None, test=False):
        argument_objs = self._build_arguments()

        task_name, arguments = ComplexArguments(argument_objs).parse(args)
        if test: arguments.model_args.unit_test = test
        log.info("Start Example Task:{}, with arguments:{}".format(task_name, to_json_string(arguments)))
        task_action = arguments.action
        task_instance = self.TASKS[task_name](arguments)

        eval('task_instance.' + task_action + '()')
