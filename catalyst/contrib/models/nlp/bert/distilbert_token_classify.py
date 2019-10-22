import torch.nn as nn

from transformers.modeling_distilbert import DistilBertPreTrainedModel, \
    DistilBertModel


class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    def __init__(self, config, num_classes=None):
        super().__init__(config)

        self.bert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

        self.init_weights()

    def forward(self, features, attention_mask=None, head_mask=None):
        outputs = self.bert(
            features, attention_mask=attention_mask, head_mask=head_mask
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = logits.permute(0, 2, 1)

        return logits


__all__ = ["DistilBertForTokenClassification"]
