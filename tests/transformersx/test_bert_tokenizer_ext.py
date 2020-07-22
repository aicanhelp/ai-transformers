from transformersx.model.bert.tokenization_bert_ext import (
    _create_token_type_ids_from_sequences_for_multiple_sentences,
    _get_special_tokens_mask_for_multiple_sentences
)


class Test_BertTokenizerExt():
    def test_get_special_tokens_mask_for_multiple_sentences(self):
        token_ids_0 = [1, 0, 1, 0]
        token_ids_1 = [1, 0, 1, 0, 9, 99, 0]
        token_ids_2 = [1, 0, 1, 0, 9, 99, 0, 9, 99, 1]
        new_token_ids = _get_special_tokens_mask_for_multiple_sentences(token_ids_0, 99)
        assert new_token_ids == [1, 0, 0, 0, 0, 1]
        new_token_ids = _get_special_tokens_mask_for_multiple_sentences(token_ids_1, 99)
        assert new_token_ids == [1, 0, 0, 0, 0, 1, 1, 0, 1]
        new_token_ids = _get_special_tokens_mask_for_multiple_sentences(token_ids_2, 99)
        assert new_token_ids == [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1]

    def test_create_token_type_ids_from_sequences_for_multiple_sentences(self):
        token_ids_0 = [1, 0, 1, 0]
        token_ids_1 = [1, 0, 1, 0, 9, 99, 0]
        token_ids_2 = [1, 0, 1, 0, 9, 99, 0, 9, 99, 1]
        new_token_ids = _create_token_type_ids_from_sequences_for_multiple_sentences(token_ids_0, 99)
        assert new_token_ids == [0, 0, 0, 0, 0, 0]
        new_token_ids = _create_token_type_ids_from_sequences_for_multiple_sentences(token_ids_1, 99)
        assert new_token_ids == [0, 0, 0, 0, 0, 0, 1, 1, 1]
        new_token_ids = _create_token_type_ids_from_sequences_for_multiple_sentences(token_ids_2, 99)
        assert new_token_ids == [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
