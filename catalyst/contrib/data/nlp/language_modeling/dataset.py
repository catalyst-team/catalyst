from typing import Iterable, Union
import warnings

from tqdm.auto import tqdm

import torch
import transformers
from transformers import AutoTokenizer


class LMDataset(torch.utils.data.Dataset):
    """
    Dataset for (masked) language model task.
    Can sort sequnces for efficient padding.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: Union[
            str, transformers.tokenization_utils.PreTrainedTokenizer
        ],
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
            lazy (bool): If True then tokenize sequence
                in __getitem__ method
                else will tokenize in __init__ also
                if set to true sorting is unavialible
        """
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(
            tokenizer, transformers.tokenization_utils.PreTrainedTokenizer
        ):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "tokenizer argument should be model name"
                + " or huggingface pre trained tokenizer"
            )

        self.max_seq_length = max_seq_length

        self.lazy = lazy

        if lazy:
            self.texts = texts

        if not lazy:
            pbar = tqdm(texts, desc="tokenizing texts")
            self.encoded = [
                self.tokenizer.encode(text, max_len=max_seq_length)
                for text in pbar
            ]
            if sort:
                self.encoded.sort(key=len)

        elif sort:
            warnings.warn(
                "Warning! lazy set to True so we can't sort"
                + " sequences by length.\n"
                + "You should set sort=False if lazy=True"
            )
        self.length = len(texts)

    def __len__(self):
        """Return length of dataloader"""
        return len(self.length)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        If lazy is True then encode text and return
        else just return already encoded text
        """
        if not self.lazy:
            return torch.tensor(self.encoded[idx])

        encoded = self.tokenizer.encode(
            self.texts[idx], max_len=self.max_seq_length
        )
        return torch.tensor(encoded)
