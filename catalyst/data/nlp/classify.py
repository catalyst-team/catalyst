from typing import Mapping
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer


class ClassificationDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: pd.Series, max_seq_length=512):
        logging.getLogger("transformers.tokenization_utils").setLevel(
            logging.FATAL
        )

        # pandas Series with texts to classify
        self.texts = texts
        # pandas Series with classification labels (strings)
        self.labels = labels

        # {'class1': 0, 'class2': 1, 'class3': 2, ...}
        # using this instead of `sklearn.preprocessing.LabelEncoder`
        # no easily handle unknown target values
        self.label_dict = dict(zip(labels.unique(), range(labels.nunique())))
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

        x, y = self.texts[index], self.labels[index]

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

        y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)

        mask = torch.ones_like(x_encoded, dtype=int)
        mask_pad = torch.zeros_like(pad_ids, dtype=int)
        mask = torch.cat((mask, mask_pad))

        return {
            "features": x_tensor,
            "targets": y_encoded,
            'mask': mask
        }


__all__ = ["ClassificationDataset"]
