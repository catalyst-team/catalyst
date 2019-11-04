from typing import Mapping
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import numpy as np


class ClassificationDataset(Dataset):
	def __init__(self, texts: pd.Series, labels: pd.Series, max_seq_length=512, batch_size=None, use_seq_bucketing=False):
		logging.getLogger("transformers.tokenization_utils").setLevel(
			logging.FATAL
		)

		if use_seq_bucketing is True and batch_size is None:
			raise Exception("You cannot use sequence bucketing without batch_size flag")

		self.batch_size = batch_size
		self.use_seq_bucketing = use_seq_bucketing

		# pandas Series with texts to classify
		self.texts = texts
		# pandas Series with classification labels (strings)
		self.labels = labels

		# {'class1': 0, 'class2': 1, 'class3': 2, ...}
		# using this instead of `sklearn.preprocessing.LabelEncoder`
		# no easily handle unknown target values
		self.label_dict = None
		if labels is not None:
			self.label_dict = dict(zip(labels.unique(), range(labels.nunique())))
		self.max_seq_length = max_seq_length

		self.tokenizer = DistilBertTokenizer.from_pretrained(
			"distilbert-base-uncased"
		)
		self.sep_vid = self.tokenizer.vocab["[SEP]"]
		self.cls_vid = self.tokenizer.vocab["[CLS]"]
		self.pad_vid = self.tokenizer.vocab["[PAD]"]

		self.frame_with_pad = None

		if use_seq_bucketing is True:
			self.frame_with_pad = self._generate_ind_list_()

	def _generate_ind_list_(self):
		texts_length = self.texts.apply(lambda text: len(self.tokenizer.encode(
														text, 
														add_special_tokens=True, 
														max_length=self.max_seq_length,
													))
		)

		texts_length.sort_values(inplace=True, ascending=False)
		texts_length.name = "texts length"
		texts_length = texts_length.reset_index()
		if "id" in texts_length.columns:
			texts_length.columns = ['index', 'texts length']

		pad_length = []

		i = 0
		for bs in range(len(texts_length) // self.batch_size + 1):
			texts_length_batch = texts_length.iloc[i:i + self.batch_size]
			i += self.batch_size
			if texts_length_batch.empty is True:
				break
			max_len_batch = texts_length_batch['texts length'].max()
			lengths = list(texts_length_batch['texts length'])
			pad_length.extend(
				[
					max_len_batch - lengths[k]
					for k in range(len(lengths))
				]
			)
		assert len(pad_length) == len(texts_length)
		texts_length['pad length'] = pad_length
		
		return texts_length


	def __len__(self):
		return len(self.texts)

	def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

		metadata = None
		x, y = None, None
		if self.use_seq_bucketing is True:
			metadata = self.frame_with_pad.iloc[index]
			x = self.texts[metadata["index"]]
			if self.labels is not None:
				y = self.labels[metadata["index"]]
		else:
			x = self.texts[index]
			if self.labels is not None:
				y = self.labels[index]

		#TODO: remove max_length flag and use custom function for seq cropping
		# (eg. 128 head tokens, 128 tail tokens)
		x_encoded = self.tokenizer.encode(
			x,
			add_special_tokens=True,
			max_length=self.max_seq_length,
			return_tensors="pt",
		).squeeze(0)

		true_seq_length = x_encoded.size(0)

		if self.use_seq_bucketing is True:
			pad_size = metadata["pad length"]
		else:
			pad_size = self.max_seq_length - true_seq_length

		pad_ids = torch.Tensor([self.pad_vid] * pad_size).long()
		x_tensor = torch.cat((x_encoded, pad_ids))


		y_encoded = None
		if y is not None:
			y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)

		mask = torch.ones_like(x_encoded, dtype=int)
		mask_pad = torch.zeros_like(pad_ids, dtype=int)
		mask = torch.cat((mask, mask_pad))

		return_dict = {
			"features": x_tensor,
			'mask': mask
		}

		if y_encoded is not None:
			return_dict["targets"] = y_encoded
		return return_dict


__all__ = ["ClassificationDataset"]
