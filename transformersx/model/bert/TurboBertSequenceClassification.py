import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertModel
import turbo_transformers
from turbo_transformers import PoolingType
from turbo_transformers import ReturnType


class TurboBertForSequenceClassification(nn.Module):
    def __init__(self, bert, config):
        super().__init__()
        self.num_labels = config.num_labels

        self.bert = bert
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_masks=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            pooling_type=PoolingType.FIRST,
            return_type=ReturnType.TORCH
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    @staticmethod
    def from_pretrained(model_id_or_path: str, config, **kwargs):
        torch_model = BertModel.from_pretrained(model_id_or_path, config=config, **kwargs)

        turbo_model = turbo_transformers.BertModelWithPooler.from_torch(torch_model, device=torch.device('cuda:0'))
        model = TurboBertForSequenceClassification(turbo_model, config)
        return model
