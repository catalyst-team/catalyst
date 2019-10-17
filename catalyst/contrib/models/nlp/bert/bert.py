import torch.nn as nn

from catalyst.contrib.models.nlp.bert.distilbert_token_classify import \
    DistilBertForTokenClassification

MODEL_MAP = {
    "distilbert/token_class": (
        DistilBertForTokenClassification, "distilbert-base-uncased"
    ),
}


class BertModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_type = kwargs.pop("model_type", None)
        model_cls, model_base = MODEL_MAP[model_type]
        self.model = model_cls.from_pretrained(model_base, **kwargs)

    def forward(self, features, mask=None, head_mask=None):
        logits = self.model(features, attention_mask=mask, head_mask=head_mask)
        logits = logits.permute(0, 2, 1)
        return logits


__all__ = ["BertModel"]
