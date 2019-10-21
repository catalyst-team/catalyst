import torch.nn as nn

from transformers.modeling_distilbert import DistilBertPreTrainedModel, \
    DistilBertModel


class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config, num_classes=None):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, features, attention_mask=None, head_mask=None):
        distilbert_output = self.distilbert(input_ids=features,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]                  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                   # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)   # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)             # (bs, dim)
        pooled_output = self.dropout(pooled_output)          # (bs, dim)
        logits = self.classifier(pooled_output)              # (bs, dim)

        return logits


__all__ = ["DistilBertForSequenceClassification"]
