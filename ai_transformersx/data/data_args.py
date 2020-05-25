from dataclasses import field, dataclass
from ai_harness import harnessutils as aiutils

log = aiutils.getLogger('task')


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(default="",
                           metadata={
                               "help": "the task name."}
                           )

    model_mode_for_data: str = field(default="classification",
                                     metadata={"help": "the model of model: classification or regression"})

    ##todo: refactor to use the max_position_embeddings of the config of model
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    predict: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    progress_bar: bool = field(default=True, metadata={"help": "Whether shows the progress_bar"})
