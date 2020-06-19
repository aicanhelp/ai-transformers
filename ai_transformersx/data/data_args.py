from ..transformersx_base import *


@configclass
class DataArguments:
    task_name: str = field("", "the task name.")

    data_dir: str = field('', 'input the data dir for processing')
    model_mode_for_data: str = field("classification", "the model of model: classification or regression")

    ##todo: refactor to use the max_position_embeddings of the config of model
    max_seq_length: int = field(512,
                                "The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.")
    predict: bool = field(False, "Overwrite the cached training and evaluation sets")
    show_feature_num: int = field(5, "the number of features for printing after converted")
    overwrite_cache: bool = field(False, "Overwrite the cached training and evaluation sets")
    progress_bar: bool = field(default=True, metadata={"help": "Whether shows the progress_bar"})
