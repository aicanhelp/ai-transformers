import os
import time

import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm, trange
from transformers import PreTrainedTokenizer, torch_distributed_zero_first, \
    RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
from typing import List, Optional, Union

from transformers.tokenization_utils import BatchEncoding

from .configuration import DataArguments, log
from .dataprocessor import DataProcessor

from .trainer_utils import InputFeatures, InputExample


###TODO: The Dataset should be refactored for many data sources
###TODO: Behind the dataset, it should have a data sources. So the data processor should be the data sources.
class TaskDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: DataArguments
    features: List[InputFeatures]

    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            processor: DataProcessor,
            limit_length: Optional[int] = None,
            evaluate=False,
            local_rank=-1
    ):
        self.args = args
        # Load data features from cache or dataset file
        cached_features_file = None
        if not args.predict:
            cached_features_file = os.path.join(
                processor.data_dir(),
                "cached_{}_{}_{}_{}".format(
                    "dev" if evaluate else "train", tokenizer.__class__.__name__, str(args.max_seq_length),
                    args.task_name,
                ),
            )
        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if not args.predict and os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                log.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                log.info(f"Creating features from dataset file at {processor.data_dir()}")
                label_list = processor.get_labels()
                if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
                        RobertaTokenizer,
                        RobertaTokenizerFast,
                        XLMRobertaTokenizer,
                ):
                    # HACK(label indices are swapped in RoBERTa pretrained model)
                    label_list[1], label_list[2] = label_list[2], label_list[1]
                examples = (
                    processor.get_dev_examples()
                    if evaluate
                    else processor.get_train_examples()
                )
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = _glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    processor,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.args.model_mode_for_data,
                    progress_bar=args.progress_bar,
                    evaluate=evaluate
                )
                if not args.predict and local_rank in [-1, 0]:
                    log.info("Saving features into cached file %s", cached_features_file)
                    start = time.time()
                    torch.save(self.features, cached_features_file)
                    # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                    log.info(
                        f"Saved features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                    )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def _glue_convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        processor: DataProcessor,
        max_length: Optional[int] = None,
        task=None,
        label_list=None,
        output_mode=None,
        progress_bar=False,
        evaluate=False
):
    """
        Loads a data file into a list of ``InputFeatures``

        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length. Defaults to the tokenizer's max_len
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """
    log.info("Converting Examples to features .... total: " + str(len(examples)))
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        if label_list is None:
            label_list = processor.get_labels()
            log.info("Using label list %s for task %s" % (label_list, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    # log.info("1. Tokenizer encoding examples .... total: " + str(len(examples)))
    # epoch_iterator = tqdm(examples, desc="Iteration", disable=not progress_bar)
    # batch_encoding = tokenizer.batch_encode_plus(
    #     [(example.text_a, example.text_b) for example in epoch_iterator], max_length=max_length, pad_to_max_length=True,
    # )

    batch_encoding = batch_encode_plus(tokenizer, examples, max_length, progress_bar)

    log.info("2. Converting Examples to Features .... total: " + str(len(examples)))
    epoch_iterator = tqdm(examples, desc="Iteration", disable=not progress_bar)
    features = []

    for k in batch_encoding.keys():
        log.info("key={},size={}".format(k, str(len(batch_encoding[k]))))

    for i, example in enumerate(epoch_iterator):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i], guid=i if evaluate else None)

        features.append(feature)

    for i, example in enumerate(examples[:5]):
        log.info("*** Example ***")
        log.info("guid: %s" % (example.guid))
        log.info("features: %s" % features[i])

    return features


def batch_encode_plus(tokenizer, examples, max_length, progress_bar=False):
    log.info("1. Tokenizer encoding examples .... total: " + str(len(examples)))
    total = len(examples)
    epoch_iterator = tqdm(range(0, total, 100), desc="Iteration", disable=not progress_bar)

    batch_outputs = {}
    for step in epoch_iterator:
        batch_encoding = tokenizer.batch_encode_plus(
            [(example.text_a, example.text_b) for example in examples[step:step + 100]], max_length=max_length,
            pad_to_max_length=True,
        )

        for key, value in batch_encoding.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].extend(value)

    return BatchEncoding(batch_outputs)
