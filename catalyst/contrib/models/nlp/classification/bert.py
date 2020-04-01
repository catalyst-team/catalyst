from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class BertClassifier(nn.Module):
    """Simplified version of the same class by HuggingFace.

    See ``transformers/modeling_distilbert.py`` in the transformers repository.
    """

    def __init__(
        self, pretrained_model_name: str, num_classes: Optional[int] = None
    ):
        """
        Args:
            pretrained_model_name (str): HuggingFace model name.
                See transformers/modeling_auto.py
            num_classes (int, optional): the number of class labels
                in the classification task
        """
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )

        self.distilbert = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        )
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.seq_classif_dropout),
            nn.Linear(config.dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute class probabilities for the input sequence.

        Args:
            features (torch.Tensor): ids of each token,
                size ([bs, seq_length]
            attention_mask (torch.Tensor, optional): binary tensor,
                used to select tokens which are used to compute attention
                scores in the self-attention heads, size [bs, seq_length]
            head_mask (torch.Tensor, optional): 1.0 in head_mask indicates that
                we keep the head, size: [num_heads]
                or [num_hidden_layers x num_heads]

        Returns:
            PyTorch Tensor with predicted class probabilities
        """
        assert attention_mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(
            input_ids=features,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # we only need the hidden state here and don't need
        # transformer output, so index 0
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        # we take embeddings from the [CLS] token, so again index 0
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits


__all__ = ["BertClassifier"]
