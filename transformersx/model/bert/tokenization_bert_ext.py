from typing import Union, List, Optional

from transformers import BertTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, TensorType, BatchEncoding


def __get_token_sep_indexes(token_ids, sep_id):
    sep_indexes = []
    for i, v in enumerate(token_ids):
        if v == sep_id: sep_indexes.append(i)
    return sep_indexes


def _get_special_tokens_mask_for_multiple_sentences(token_ids_0: List[int], sep_id) -> List[int]:
    sep_indexes = __get_token_sep_indexes(token_ids_0, sep_id)
    tokens = [0] * len(token_ids_0)

    for i in sep_indexes:
        tokens[i - 1] = 1
        tokens[i] = 1
    return [1] + tokens + [1]


def _create_token_type_ids_from_sequences_for_multiple_sentences(token_ids_0: List[int], sep_id) -> List[int]:
    sep_indexes = __get_token_sep_indexes(token_ids_0, sep_id)
    if len(sep_indexes) == 0: return (2 + len(token_ids_0)) * [0]
    sep_indexes.append(len(token_ids_0) + 1)
    sep_indexes = [c - sep_indexes[i - 1] if i > 0 else c for i, c in enumerate(sep_indexes)]
    sep_indexes[0] = sep_indexes[0] + 1

    tokens = []
    for i in sep_indexes: tokens = tokens + [i & 0b0001] * i

    return tokens


class BertTokenizerExt(BertTokenizer):
    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            **kwargs
    ):
        super().__init__(vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token,
                         cls_token, mask_token, tokenize_chinese_chars, **kwargs)

    def __call__(
            self,
            text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
            text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str] = False,
            truncation: Union[bool, str] = False,
            max_length: Optional[int] = None,
            stride: int = 0,
            is_pretokenized: bool = False,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            multiple_sentences: bool = False,
            **kwargs
    ) -> BatchEncoding:
        self.multiple_sentences = multiple_sentences
        if multiple_sentences:
            assert not text_pair, 'For multiple sentences, all sentences must be put into the first parameter.'
            if isinstance(text, (tuple, list)) and len(text) > 1:
                text = (' ' + self.sep_token + ' ' + self.cls_token + ' ').join(text)

        return super().__call__(text, text_pair, add_special_tokens, padding, truncation, max_length, stride,
                                is_pretokenized, pad_to_multiple_of, return_tensors, return_token_type_ids,
                                return_attention_mask, return_overflowing_tokens, return_special_tokens_mask,
                                return_offsets_mapping, return_length, verbose,
                                **kwargs)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if self.multiple_sentences:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return super().build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        if self.multiple_sentences:
            return _get_special_tokens_mask_for_multiple_sentences(token_ids_0, self.sep_token_id)
        return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens)

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        if self.multiple_sentences: return _create_token_type_ids_from_sequences_for_multiple_sentences(
            token_ids_0, self.sep_token_id)

        return super().create_token_type_ids_from_sequences(token_ids_0, token_ids_1)
