import unittest

import numpy as np
from numpy.testing import assert_array_equal

from catalyst.data.nlp.classify import ClassificationDataset

from torch.utils.data import DataLoader

import pandas as pd

texts = pd.Series([
    "Ryan is great at researching new technologies ",
    "and companies and providing the right kind of analysis for PM",
    "I think you wanna be a research scientist",
    "An important concept in the paper is Back-Translation, in which a sentence is translated to the target language and back to the source.",
    "This is a common pre-processing algorithm and a summary of it can be found here.",
    "Each training sample consists of the same text in two languages, whereas in BERT each sample is built from a single language.",
    "The new metadata helps the model learn the relationship between related tokens in different languages."
])

classes = pd.Series(['a', 'a', 'a', 'b', 'a', 'a', 'b'])

"""
frame_with_pad table in dataset for this texts

   index  texts length  pad length
0      3            29           0
1      5            26           3
2      4            20           9
3      6            19           0
4      1            13           6
5      2            10           9
6      0             9           0

"""

class ClassificationDatasetTests(unittest.TestCase):
	def test_seq_bucketing(self):
		dataset = ClassificationDataset(texts, classes, batch_size=3, use_seq_bucketing=True)
		data, tg = dataset[0]["features"], dataset[0]["targets"]
		self.assertEqual(1, tg.item())

	def test_batches(self):
		dataset = ClassificationDataset(texts, classes, batch_size=3, use_seq_bucketing=True)
		loader = DataLoader(dataset, batch_size=3)
		shapes = [29, 19, 9]

		ans = []
		for batch in loader:
			ans.append(batch['features'].shape[1])
		self.assertEqual(shapes, ans)

	def test_dataset(self):
		dataset = ClassificationDataset(texts, classes)
		data, tg = dataset[0]["features"], dataset[0]["targets"]
		self.assertEqual(0, tg.item())

	def test_dataset_without_labels(self):
		dataset = ClassificationDataset(texts, None)
		data = dataset[0]["features"]
		flag = "targets" not in dataset[0].keys()
		self.assertEqual(flag, True)

	def test_batches(self):
		dataset = ClassificationDataset(texts, classes)
		loader = DataLoader(dataset, batch_size=3)
		shapes = [512, 512, 512]

		ans = []
		for batch in loader:
			ans.append(batch['features'].shape[1])
		self.assertEqual(shapes, ans)