import unittest

from catalyst.data.nlp.classify import TextClassificationDataset

texts = [
    "The color of this T-shirt is sooo so horrible",
    "Nice gears, enjoy price-ti-quality ratio"
]

labels = ["negative", "positive"]


class TextClassificationDatasetTests(unittest.TestCase):

    def test_should_have_cls_id_as_first_token_for_input_ids(self):
        dataset = TextClassificationDataset(texts, labels)
        features = dataset[0]["features"]
        self.assertEqual(dataset.cls_vid, features[0])

    def test_input_ids_should_be_padded(self):
        dataset = TextClassificationDataset(texts, labels)
        features = dataset[0]["features"]
        self.assertEqual(512, features.size(0))

    def test_mask_sum_should_be_eq_to_seq_len(self):
        dataset = TextClassificationDataset(texts, labels)
        mask = dataset[0]["attention_mask"]
        self.assertEqual(512, mask.size(0))
        self.assertEqual(14, mask.sum())
        self.assertEqual(14, mask[:14].sum())

    def test_label_dict(self):
        dataset = TextClassificationDataset(texts, labels)
        label_dict = dataset.label_dict
        self.assertEqual({"negative": 0, "positive": 1}, label_dict)


__all__ = ["TextClassificationDatasetTests"]
