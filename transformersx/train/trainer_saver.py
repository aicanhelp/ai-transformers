from .trainer_base import *
from .training_args import TaskTrainingArguments
from .trainer_optimizers import TaskTrainerOptimizers


class TaskTrainerSaver(TrainerBase):
    def __init__(self, args: TaskTrainingArguments,
                 model: PreTrainedModel):
        super().__init__(args)
        self._model = model

    def save_check_point(self, trainer_optimizers: TaskTrainerOptimizers, global_step):
        if self._args.save_steps > 0 and global_step % self._args.save_steps == 0:
            # In all cases (even distributed/parallel), self.model is always a reference
            # to the model we want to save.
            model = trainer_optimizers.get_model()
            if hasattr(model, "module"):
                assert model.module is self._model
            else:
                assert model is self._model
            # Save model checkpoint
            output_dir = os.path.join(
                self._args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}"
            )

            self._save_check_point_model(output_dir)
            self._rotate_checkpoints()

            torch.save(trainer_optimizers.get_optimizer().state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(trainer_optimizers.get_scheduler().state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

    def load_check_point(self, model_path, trainer_optimizers: TaskTrainerOptimizers):
        if (
                model_path is not None
                and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
                and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            logger.info("Loading optimizer and scheduler from :" + model_path)
            trainer_optimizers.get_optimizer().load_state_dict(torch.load(os.path.join(model_path, "optimizer.pt")))
            trainer_optimizers.get_scheduler().load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self._args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self._args.save_total_limit is None or self._args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self._args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self._args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def _save_check_point_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the master process.
        """

        if self.is_tpu_available():
            self._save_check_point_tpu(output_dir)

        elif self.is_world_master():
            self._save_check_point(output_dir)

    def _save_check_point_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self._args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if self.xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self._args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self._model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        self.xm.rendezvous("saving_checkpoint")
        self._model.save_pretrained(output_dir)

    def _save_check_point(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self._args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self._model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self._model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self._args, os.path.join(output_dir, "training_args.bin"))
