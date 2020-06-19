from typing import Dict, NamedTuple, Optional, List, Union
import numpy as np
import torch
from dataclasses import dataclass
from transformers import DataCollator
from transformers.data.data_collator import InputDataClass


class TaskEvalPrediction(NamedTuple):
    predictions: np.ndarray
    label_ids: np.ndarray


class TaskPredictionOutput(NamedTuple):
    guids: np.ndarray
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    eval_losses: np.ndarray
    metrics: Optional[Dict[str, float]] = None


class TaskTrainOutput(NamedTuple):
    global_step: int
    training_loss: float


@dataclass
class TaskDefaultDataCollatorx(DataCollator):
    def collate_batch(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        first = features[0]

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if hasattr(first, "label") and first.label is not None:
            if type(first.label) is int:
                labels = torch.tensor([f.label for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label for f in features], dtype=torch.float)
            batch = {"labels": labels}
        elif hasattr(first, "label_ids") and first.label_ids is not None:
            if type(first.label_ids[0]) is int:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f.label_ids for f in features], dtype=torch.float)
            batch = {"labels": labels}
        elif hasattr(first, 'guid') and first.guid is not None:
            guids = torch.tensor([f.guid for f in features], dtype=torch.long)
            batch = {"guid": guids}
        else:
            batch = {}

        # Handling of all other possible attributes.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in vars(first).items():
            if k not in ("guid", "label", "label_ids") and v is not None and not isinstance(v, str):
                batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        return batch


PREFIX_CHECKPOINT_DIR = "checkpoint"
