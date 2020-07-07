import json

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel

from .trainer_base import *
import os
from .trainer_scheduler import TaskTrainedScheduler

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn(
            "W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


@configclass
class TrainerLoggerConfig():
    logging_dir: str = field(None, "Tensorboard log dir.")
    logging_first_step: bool = field(False, "Log and eval the first global_step")
    logging_steps: int = field(500, "Log every X updates steps.")
    tpu_metrics_debug: bool = field(False, "TPU: Whether to print debug metrics")


class TaskTrainerLogger():
    def __init__(self, trainer_env: TrainerEnv,
                 model: PreTrainedModel,
                 tb_writer: Optional["SummaryWriter"] = None):
        self._env = trainer_env
        self.config: TrainerLoggerConfig = trainer_env.get_config(TrainerLoggerConfig)
        self._model = model
        self._tb_writer = tb_writer

    def __init_logger(self, tb_writer):
        if self._tb_writer is not None:
            self._tb_writer = tb_writer
        elif is_tensorboard_available() and self._env.config.local_rank in [-1, 0]:
            self._tb_writer = SummaryWriter(log_dir=self.config.logging_dir)
        if not is_tensorboard_available():
            log.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            log.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        log.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self._env.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self._model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.config.logging_steps)
            )

    def log_pre_train(self, train_dataloader, train_scheduler: TaskTrainedScheduler):
        if self._tb_writer is not None:
            self._tb_writer.add_text("args", self._env.args.to_json_string())
            self._tb_writer.add_hparams(self._env.args.to_sanitized_dict(), metric_dict={})

        log.info("***** Running training *****")
        log.info("  Num examples = %d", self._env.num_examples(train_dataloader))
        log.info("  Num Epochs = %d", train_scheduler.num_train_epochs)
        log.info("  Instantaneous batch size per device = %d", self._env.config.per_gpu_train_batch_size)
        log.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                 train_scheduler.total_train_batch_size)
        log.info("  Gradient Accumulation steps = %d", train_scheduler.config.gradient_accumulation_steps)
        log.info("  Total optimization steps = %d", train_scheduler.t_total)

    def is_need_log_step(self, global_step):
        return (self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0) or (
                global_step == 1 and self.config.logging_first_step)

    def log_train_step(self, epoch,
                       global_step,
                       logs: Dict[str, float],
                       iterator: Optional[tqdm] = None) -> None:
        if epoch is not None:
            logs["epoch"] = epoch
        if self._tb_writer:
            for k, v in logs.items():
                self._tb_writer.add_scalar(k, v, global_step)
        if is_wandb_available():
            wandb.log(logs, step=global_step)
        output = json.dumps({**logs, **{"step": global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def log_train_epoch(self):
        if self.config.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            self._env.tpu_metrics()

    def log_train_end(self):
        if self._tb_writer:
            self._tb_writer.close()

        log.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
