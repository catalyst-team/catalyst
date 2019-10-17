from typing import Mapping
import logging

import torch
from torch.utils.data import Dataset

from transformers import DistilBertTokenizer


class KeyPhrasesDataset(Dataset):
    def __init__(self, texts, keyphrases, max_seq_length=512):
        logging.getLogger("transformers.tokenization_utils").setLevel(
            logging.FATAL
        )

        self.texts = texts
        self.keyphrases = keyphrases
        self.max_seq_length = max_seq_length

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        def _find_inclusions(list_a, list_b):
            return [
                x for x in range(len(list_a))
                if list_a[x:x + len(list_b)] == list_b
            ]

        x, y = self.texts[index], self.keyphrases[index]

        x_encoded = self.tokenizer.encode(
            x,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).squeeze(0)

        true_seq_length = x_encoded.size(0)
        pad_size = self.max_seq_length - true_seq_length

        pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
        x_tensor = torch.cat((x_encoded, pad_ids))

        mask = torch.ones_like(x_encoded, dtype=int)
        mask_pad = torch.zeros_like(pad_ids, dtype=int)
        mask = torch.cat((mask, mask_pad))

        x_list = x_encoded.tolist()
        y_encoded = [self.tokenizer.encode(x) for x in y]
        y_start_pos = [_find_inclusions(x_list, y) for y in y_encoded]
        y_positions = [
            list(range(x[0], x[0] + len(y)))
            for x, y in zip(y_start_pos, y_encoded) if x
        ]
        y_positions = [item for sublist in y_positions for item in sublist]

        labels = torch.zeros_like(x_tensor)
        labels[y_positions] = 1

        return {
            "features": x_tensor,
            "targets": labels,
            "mask": mask
        }


__all__ = ["KeyPhrasesDataset"]
