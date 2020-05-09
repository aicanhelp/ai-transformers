from aiharness import harnessutils as utils
from transformers.modeling_auto import MODEL_FOR_MULTIPLE_CHOICE_MAPPING, \
    MODEL_FOR_PRETRAINING_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_MAPPING, MODEL_WITH_LM_HEAD_MAPPING

log = utils.getLogger('data')

all_model_mappings = [MODEL_MAPPING,
                      MODEL_WITH_LM_HEAD_MAPPING,
                      MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
                      MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                      MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                      MODEL_FOR_PRETRAINING_MAPPING,
                      MODEL_FOR_MULTIPLE_CHOICE_MAPPING
                      ]


def model_class(config):
    for mapping in all_model_mappings:
        for config_class, model_class in mapping.items():
            if isinstance(config, config_class):
                return model_class
