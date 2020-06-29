from ..model import ALL_TASK_MODEL_PATHS
from ..transformersx_base import *


@configclass
class FastTaskArguments:
    action: str = field("train", "the task action: train,eval,predict")
    data_dir: str = field("/app/dataset", "the data directory for train,eval,test,label")
    model_finetuned_dir: str = field("/app/models/finetuning",
                                     "The 'model_base_dir' generally is for the finetuned models."
                                     "Generally, the model is loaded from 'model_finetuned_dir' firstly."
                                     " If the model cannot be found, it will be loaded from the model_pretrained_dir.")

    model_pretrained_dir: str = field("/app/models/pretrained",
                                      "This folder is for the pretrained models downloaded from Internet.")
    model_name: str = field("bert-base-chinese", "the name of model: " + str(ALL_TASK_MODEL_PATHS))
    train_file: str = field('train.csv', "the train data file name")
    eval_file: str = field('dev.csv', "the eval data file name")
    labels_file: str = field('labels.csv', "the test data file name")
    text_col: str = field('text', '')
    label_col: str = field('label', '')
    batch_size_per_gpu: int = field(16, '')
    learning_rate: float = field(5e-5, "The initial learning rate for Adam.")
    num_train_epochs: int = field(3, "Total number of training epochs to perform.")
    max_seq_length: int = field(512, '')
    multi_label: bool = field(False, '')
    model_type: str = field('bert', '')
    fp16: bool = field(False, '')
