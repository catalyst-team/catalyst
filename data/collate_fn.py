import collections

import torch
from torch.utils.data.dataloader import default_collate


def text_collate_fn(data, txt_key="txt", len_key="len"):
    """
    Creates mini-batch tensors from the list of tuples (text, whatever).

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (ch, h, w).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        text: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order)
    data.sort(key=lambda x: len(x[txt_key]), reverse=True)
    text_original = list(map(lambda x: x[txt_key], data))

    # Merge other data (from tuple of 3D tensor to 4D tensor).
    def stack_non_text_features(data, key):
        features = list(map(lambda x: x[key], data))
        features = torch.stack(features, 0)
        return features

    # too hacky
    other_data = {
        key: stack_non_text_features(data, key)
        for key in data[0] if key != txt_key}

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in text_original]
    text_processed = torch.zeros(len(text_original), max(lengths)).long()
    for i, text_row in enumerate(text_original):
        end = lengths[i]
        text_processed[i, :end] = text_row[:end]

    text_data = {
        txt_key: text_processed,
        len_key: lengths
    }

    data = {
        **other_data,
        **text_data
    }

    return data


class FilteringCollateFn:
    """
    Callable object doing job of collate_fn like default_collate, but does not
    cast batch items with specified key to torch.Tensor. Only adds them to list.
    Supports only key-value format batches
    """

    def __init__(self, *keys):
        """
        :param keys: Keys having values that will not be
            converted to tensor and stacked
        """

        self.keys = keys

    def __call__(self, batch):
        if isinstance(batch[0], collections.Mapping):
            result = {}
            for key in batch[0]:
                items = [d[key] for d in batch]
                if key not in self.keys:
                    items = default_collate(items)
                result[key] = items
            return result
        else:
            return default_collate(batch)
