from catalyst.data.nlp import TextClassificationDataset

texts = [
    "The color of this T-shirt is sooo so horrible",
    "Nice gears, enjoy price-ti-quality ratio",
]

labels = ["negative", "positive"]


def test_cls_id_as_first_token_for_input_ids():
    """@TODO: Docs. Contribution is welcome."""
    dataset = TextClassificationDataset(texts, labels)
    features = dataset[0]["features"]
    assert features[0] == dataset.cls_vid


def test_input_ids_are_padded():
    """@TODO: Docs. Contribution is welcome."""
    dataset = TextClassificationDataset(texts, labels)
    features = dataset[0]["features"]
    assert features.size(0) == 512


def test_mask_sum_eq_to_seq_len():
    """@TODO: Docs. Contribution is welcome."""
    dataset = TextClassificationDataset(texts, labels)
    mask = dataset[0]["attention_mask"]
    assert mask.size(0) == 512
    assert mask.sum() == 14
    assert mask[:14].sum() == 14


def test_label_dict():
    """@TODO: Docs. Contribution is welcome."""
    dataset = TextClassificationDataset(texts, labels)
    label_dict = dataset.label_dict
    assert label_dict == {"negative": 0, "positive": 1}
