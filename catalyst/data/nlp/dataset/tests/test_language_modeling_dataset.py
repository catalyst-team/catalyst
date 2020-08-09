import pytest

import torch  # noqa: F401
from transformers import AutoTokenizer

from catalyst.data.nlp import LanguageModelingDataset

texts = [
    "Bonaparte Crossing the Alps is an oil-on-canvas painting by French artist",  # noqa: E501
    "Bhaskara's Lemma is an identity used as a lemma during the chakravala method. ",  # noqa: E501
]


def test_tokenizer_str():
    """Test initialization with string"""
    dataset = LanguageModelingDataset(texts, "bert-base-uncased")
    assert dataset[0] is not None
    assert len(dataset) == 2


def test_tokenizer_tokenizer():
    """Test initialization with tokenizer"""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = LanguageModelingDataset(texts, tok)
    assert dataset[0] is not None
    assert len(dataset) == 2


@pytest.mark.xfail(raises=Exception)
def test_exception_with_sort():
    """Test lazy=True sort=True case"""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = LanguageModelingDataset(  # noqa: F841
        texts, tok, lazy=True, sort=True
    )


@pytest.mark.xfail(raises=TypeError)
def test_tokenizer_type_error():
    """Test if tokenizer neither hf nor string"""
    tok = lambda x: x
    dataset = LanguageModelingDataset(texts, tok)  # noqa: F841
