import unittest

import numpy as np
from numpy.testing import assert_array_equal

from catalyst.data.nlp.key_phrase import KeyPhrasesDataset

texts = [
    "Ryan is great at researching new technologies "
    "and companies and providing the right kind of analysis for PM",
]
keyphrases = [
    ["great at researching", "right kind of analysis"],
]


class KeyPhrasesDatasetTests(unittest.TestCase):
    def test_should_have_cls_id_as_first_token_for_input_ids(self):
        dataset = KeyPhrasesDataset(texts, keyphrases)
        features = dataset[0]["features"]
        self.assertEqual(dataset.cls_vid, features[0])

    def test_input_ids_should_be_padded(self):
        dataset = KeyPhrasesDataset(texts, keyphrases)
        features = dataset[0]["features"]
        self.assertEqual(512, features.size(0))

    def test_short_sequence_mask_should_mark_meaningful_positions(self):
        dataset = KeyPhrasesDataset(texts, keyphrases)
        mask = dataset[0]["mask"]
        self.assertEqual(512, mask.size(0))
        self.assertEqual(20, mask.sum())
        self.assertEqual(20, mask[:20].sum())

    def test_labels_should_be_assigned_correctly(self):
        dataset = KeyPhrasesDataset(texts, keyphrases)
        targets = dataset[0]["targets"]

        expected = np.zeros(20, dtype=int)
        expected[[3, 4, 5]] = 1
        expected[[13, 14, 15, 16]] = 1
        assert_array_equal(expected, targets[:20].numpy())

    def test_labels_do_not_match_should_not_assign(self):
        dataset = KeyPhrasesDataset(texts, [["bla"]])
        targets = dataset[0]["targets"]
        self.assertEqual(0, targets.sum())

    def test_labels_match_partially_should_not_assign(self):
        dataset = KeyPhrasesDataset(
            texts, [["researching new amazing technologies"]]
        )
        targets = dataset[0]["targets"]
        self.assertEqual(0, targets.sum())

    def test_long_sequences_should_get_truncated(self):
        long_texts = [t * 100 for t in texts]
        dataset = KeyPhrasesDataset(long_texts, keyphrases)
        features = dataset[0]["features"]

        self.assertEqual(512, features.size(0))


__all__ = ["KeyPhrasesDatasetTests"]
