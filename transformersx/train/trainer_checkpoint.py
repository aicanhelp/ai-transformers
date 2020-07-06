from transformers import PreTrainedModel
import os
from pathlib import Path
import shutil
import re
from .trainer_base import *
from .trainer_optimizers import TaskTrainerOptimizers


@configclass
class TrainerCheckpointConfig():
    save_steps: int = field(500, "Save checkpoint every X updates steps.")
    checkpoint_base_dir: str = field('/app/models/finetuning',
                                     "The output directory where the model predictions and checkpoints will be written.")
    use_mtime: bool = field(False, 'Whether using mtime in checkpoint')
    checkpoint_prefix: str = field('checkpoint', 'the prefix of checkpoint dir')
    save_total_limit: int = field(None, "Limit the total amount of checkpoints."
                                        "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints")


class TaskTrainerCheckpoint():
    def __init__(self, trainer_env: TrainerEnv,
                 checkpoint_dir):
        self._env = trainer_env
        self.checkpoint_dir = checkpoint_dir

    def save(self, in_model, optimizers: TaskTrainerOptimizers):
        model = optimizers.get_model()
        if hasattr(model, "module"):
            assert model.module is in_model
        else:
            assert model is in_model

        self._save_check_point_model(in_model, self.checkpoint_dir)
        torch.save(optimizers.get_optimizer().state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pt"))
        torch.save(optimizers.get_scheduler().state_dict(), os.path.join(self.checkpoint_dir, "scheduler.pt"))
        log.info("Saving optimizer and scheduler states to %s", self.checkpoint_dir)
        return self

    def load(self, optimizers: TaskTrainerOptimizers):
        if (os.path.isfile(os.path.join(self.checkpoint_dir, "optimizer.pt"))
                and os.path.isfile(os.path.join(self.checkpoint_dir, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            log.info("Loading optimizer and scheduler from :" + self.checkpoint_dir)
            optimizers.get_optimizer().load_state_dict(
                torch.load(os.path.join(self.checkpoint_dir, "optimizer.pt")))
            optimizers.get_scheduler().load_state_dict(
                torch.load(os.path.join(self.checkpoint_dir, "scheduler.pt")))
        return self

    def checkpoint_number(self):
        return int(self.checkpoint_dir.split("-")[-1].split("/")[0])

    def _save_check_point_model(self, model, output_dir: Optional[str] = None):
        os.makedirs(output_dir, exist_ok=True)

        if self._env.is_tpu_available():
            self._save_check_point_tpu(model, output_dir)

        elif self._env.is_world_master():
            self._save_check_point(model, output_dir)

    def _save_check_point_tpu(self, model, output_dir: Optional[str] = None):
        log.info("Saving model checkpoint to %s", output_dir)

        if self._env.is_master_ordinal():
            torch.save(self._env.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        assert isinstance(model, PreTrainedModel), "Trainer.model appears to not be a PreTrainedModel"

        self._env.xm_rendezvous("saving_checkpoint")
        model.save_pretrained(output_dir)

    def _save_check_point(self, model, output_dir: Optional[str] = None):
        log.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        assert isinstance(model, PreTrainedModel), "Trainer.model appears to not be a PreTrainedModel"

        model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self._env.args, os.path.join(output_dir, "training_args.bin"))


class TaskTrainerCheckpointer():
    def __init__(self, task_name, trainer_env: TrainerEnv, config: TrainerCheckpointConfig):
        self._task_name = task_name
        self.config = config
        self._env = trainer_env
        self._model, self._optimizers = None, None

    def init_for_train(self, model: PreTrainedModel, trainer_optimizers: TaskTrainerOptimizers):
        self._model = model
        self._optimizers = trainer_optimizers

    def __make_checkpoint_dir(self, global_step):
        return join_path(
            self.config.checkpoint_base_dir, f"{self.config.checkpoint_prefix}-{global_step}"
        )

    def find_latest_checkpoint(self):
        checkpoints_sorted = self._sorted_checkpoints()
        if checkpoints_sorted:
            return TaskTrainerCheckpoint(self._env, join_path(self.config.checkpoint_base_dir, checkpoints_sorted[-1]))
        return None

    def save_check_point(self, global_step):
        assert self._model, 'init_for_train must be called before saving checkpoint'

        if self.config.save_steps > 0 and global_step % self.config.save_steps == 0:
            TaskTrainerCheckpoint(self._env,
                                  self.__make_checkpoint_dir(global_step)
                                  ).save(self._model, self._optimizers)
            self._rotate_checkpoints()

    def _sorted_checkpoints(self) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in
                            Path(self.config.checkpoint_base_dir).glob(f"{self.config.checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if self.config.use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{self.config.checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self) -> None:
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints()
        if len(checkpoints_sorted) <= self.config.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.config.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            log.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)
