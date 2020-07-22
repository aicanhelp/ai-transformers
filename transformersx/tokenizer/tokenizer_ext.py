from typing import Union, List, Optional

from torch._C import TensorType
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput, BatchEncoding


def multiple_sentence_encode_plus(
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
        **kwargs
) -> BatchEncoding:
    pass


def wrap_tokenizer(tokenizer: PreTrainedTokenizer):
    old_caller = tokenizer.__call__

    def new_caller(
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
            **kwargs
    ) -> BatchEncoding:
        pass

    tokenizer.__call__ = new_caller
    return tokenizer
