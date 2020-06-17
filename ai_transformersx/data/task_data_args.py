from ..transformersx_base import *


@configclass
class DataArguments:
    data_dir: str = field('/app/dataset/datastore', 'input the data dir for processing')
    overwrite: bool = field(False, "Overwrite the cached training and evaluation sets")
    progress_bar: bool = field(default=True, metadata={"help": "Whether shows the progress_bar"})
