from typing import Iterable, Union

from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizer


class LanguageModelingDataset(Dataset):
    """
    Dataset for (masked) language model task.
    Can sort sequnces for efficient padding.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: Union[str, PreTrainedTokenizer],
        max_seq_length: int = None,
        sort: bool = True,
        lazy: bool = False,
    ):
        """
        Args:
            texts (Iterable): Iterable object with text
            tokenizer (str or tokenizer): pre trained
                huggingface tokenizer or model name
            max_seq_length (int): max sequence length to tokenize
            sort (bool): If True then sort all sequences by length
                for efficient padding
            lazy (bool): If True then tokenize and encode sequence
                in __getitem__ method
                else will tokenize in __init__ also
                if set to true sorting is unavialible
        """
        if sort and lazy:
            raise Exception(
                "lazy is set to True so we can't sort"
                " sequences by length.\n"
                "You should set sort=False and lazy=True"
                " if you want to encode text in __get_item__ function"
            )
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(
            tokenizer, transformers.tokenization_utils.PreTrainedTokenizer
        ):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "tokenizer argument should be a model name"
                + " or huggingface PreTrainedTokenizer"
            )

        self.max_seq_length = max_seq_length

        self.lazy = lazy

        if lazy:
            self.texts = texts

        if not lazy:
            pbar = tqdm(texts, desc="tokenizing texts")
            self.encoded = [
                self.tokenizer.encode(text, max_length=max_seq_length)
                for text in pbar
            ]
            if sort:
                self.encoded.sort(key=len)

        self.length = len(texts)

        self._getitem_fn = (
            self._getitem_lazy if lazy else self._getitem_encoded
        )

    def __len__(self):
        """Return length of dataloader"""
        return self.length

    def _getitem_encoded(self, idx) -> torch.Tensor:
        return torch.tensor(self.encoded[idx])

    def _getitem_lazy(self, idx) -> torch.Tensor:
        encoded = self.tokenizer.encode(
            self.texts[idx], max_length=self.max_seq_length
        )
        return torch.tensor(encoded)

    def __getitem__(self, idx):
        """Return tokenized and encoded sequence"""
        return self._getitem_fn(idx)


__all__ = ["LanguageModelingDataset"]
