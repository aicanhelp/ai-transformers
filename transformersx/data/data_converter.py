from typing import Optional, Union

from transformers.tokenization_utils import BatchEncoding

from .data_models import TaskInputExample, TaskInputFeatures
from transformers import PreTrainedTokenizer
from ..transformersx_base import log
from tqdm.auto import tqdm, trange


class TaskDataConverter:
    def convert(self, examples):
        raise NotImplementedError()


class DefaultTaskDataConverter(TaskDataConverter):
    def __init__(self, tokenizer: PreTrainedTokenizer, label_list, max_length: Optional[int] = None,
                 progress_bar=False):
        self._tokenizer = tokenizer
        self._progress_bar = progress_bar
        self._max_length = max_length if max_length else tokenizer.max_len
        if self._max_length > 1024: self._max_length = 512
        self._label_list = label_list
        self._label_map = {label: i for i, label in enumerate(self._label_list)} if label_list else None

    def _label_from_example(self, example: TaskInputExample) -> Union[int, float]:
        return self._label_map[example.label] if self._label_map else float(example.label)

    def _batch_encode_plus(self, examples):
        log.info("1. Tokenizer encoding examples .... total: " + str(len(examples)))
        total = len(examples)
        epoch_iterator = tqdm(range(0, total, 100), desc="Iteration", disable=not self._progress_bar)

        batch_outputs = {}
        for step in epoch_iterator:
            batch_encoding = self._tokenizer.batch_encode_plus(
                [(example.text_a, example.text_b) for example in examples[step:step + 100]],
                max_length=self._max_length,
                pad_to_max_length=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )

            for key, value in batch_encoding.items():
                if key not in batch_outputs: batch_outputs[key] = []
                batch_outputs[key].extend(value)

        return BatchEncoding(batch_outputs)

    def _convert_features(self, examples, labels, batch_encoding):
        log.info("2. Converting Examples to Features .... total: " + str(len(examples)))
        epoch_iterator = tqdm(examples, desc="Iteration", disable=not self._progress_bar)
        features = []

        for k in batch_encoding.keys():
            log.info("key={},size={}".format(k, str(len(batch_encoding[k]))))

        for i, example in enumerate(epoch_iterator):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            feature = TaskInputFeatures(**inputs, label=labels[i] if labels else None, guid=i if not labels else None)
            features.append(feature)
        return features

    def _batch_encode_plus_list(self, examples):
        return None

    def convert(self, examples):
        has_label = (examples[0].label is not None)
        labels = [self._label_from_example(example) for example in examples] if has_label else None
        if isinstance(examples[0].text_a, str):
            batch_encoding = self._batch_encode_plus(examples)
        else:
            batch_encoding = self._batch_encode_plus_list(examples)

        features = self._convert_features(examples, labels, batch_encoding)

        for i, example in enumerate(examples[:5]):
            log.info("*** Example ***")
            log.info("guid: %s" % (example.guid))
            log.info("features: %s" % features[i])

        return features
