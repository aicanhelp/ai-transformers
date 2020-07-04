from transformersx import TransformerTask, ComplexArguments, log, configclass, field


@configclass()
class WorkspaceConfig:
    workspace_dir: str = field('/app/workspace', 'Transformers workspace directory')


class TransformersWorkspace:
    TASKS = dict()

    def __init__(self, config: WorkspaceConfig):
        self.config = config

    def register_task(self, task_name, task_class):
        assert not self.TASKS.get(task_name), "task name {} already exists, please use other name.".format(task_name)

        self.TASKS.setdefault(task_name, task_class)
        assert issubclass(task_class, (TransformerTask)), "task_class must be the subclass of TransformerTask"

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

    def start_task(self, args=None, test=False):
        argument_objs = self._build_arguments()

        task_name, arguments = ComplexArguments(argument_objs).parse(args)
        if test: arguments.model_args.unit_test = test
        log.info("Start Example Task:{}, with arguments:{}".format(task_name, str(arguments)))
        task_action = arguments.action
        task_instance = self.TASKS[task_name](arguments)

        eval('task_instance.' + task_action + '()')
