import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, model_type, num_classes=None):
        super().__init__()

        config = AutoConfig.from_pretrained(
            model_type, num_labels=num_classes)

        self.distilbert = AutoModel.from_pretrained(model_type,
                                                    config=config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, features, mask=None, head_mask=None):
        assert mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(input_ids=features,
                                            attention_mask=mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits


__all__ = ["DistilBertForSequenceClassification"]

