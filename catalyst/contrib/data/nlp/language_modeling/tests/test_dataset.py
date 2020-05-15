import torch
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling

from catalyst.contrib.data.nlp.language_modeling import LMDataset

texts = [
    """Bonaparte Crossing the Alps is an oil-on-canvas painting by French artist""",  # noqa: E501
    """Bhaskara's Lemma is an identity used as a lemma during the chakravala method. """,  # noqa: E501
]


def test_tokenizer_str():
    """Test initialization with string"""
    dataset = LMDataset(texts, "bert-base-uncased")
    assert dataset[0] is not None
    assert len(dataset) == 2


def test_tokenizer_tokenizer():
    """Test initialization with tokenizer"""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = LMDataset(texts, tok)
    assert dataset[0] is not None
    assert len(dataset) == 2


def test_collating():
    """@TODO: Docs. Contribution is welcome."""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = LMDataset(texts, tok)
    collate_fn = DataCollatorForLanguageModeling(tok).collate_batch
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=collate_fn, batch_size=2
    )
    for _batch in dataloader:
        assert True
