from typing import Dict, List  # isort:skip
import string

import numpy as np

import torch
from transformers import BertModel, BertTokenizer

from catalyst.contrib.modules import LamaPooling


def tokenize_text(
    text: str,
    tokenizer: BertTokenizer,
    max_length: int,
    strip: bool = True,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> Dict[str, np.array]:
    if strip:
        text = text.strip()
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text.replace(string.punctuation, "")

    inputs = tokenizer.encode_plus(
        text, "",
        add_special_tokens=True,
        max_length=max_length
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_mask = [1] * len(input_ids)

    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([0] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    return {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64),
    }


def process_bert_output(
    bert_output,
    bert_model: BertModel,
    pooling_groups: List[str] = None,
    mask: torch.Tensor = None,
):
    # @TODO: make this functional
    pooling = LamaPooling(
        groups=pooling_groups,
        in_features=bert_model.config.hidden_size
    ) if pooling_groups is not None else None

    def _process_features(features):
        if pooling is not None:
            features = pooling(features, mask=mask)
        return features

    output = {
        "pooling": _process_features(bert_output[0]),
        "class": bert_output[1],
    }

    if bert_model.config.output_hidden_states:
        for i, feature_ in enumerate(bert_output[2]):
            output[i] = _process_features(feature_)

    return output
