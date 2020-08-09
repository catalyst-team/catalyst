from packaging import version

import torch
from transformers import (
    __version__ as transformers_version,
    AutoModelWithLMHead,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from catalyst import dl
from catalyst.contrib.dl.callbacks import PerplexityMetricCallback
from catalyst.data.nlp import LanguageModelingDataset


class HuggingFaceRunner(dl.Runner):
    """Just an example"""

    def _handle_batch(self, batch):
        masked_lm_labels = batch.get("masked_lm_labels")
        lm_labels = batch.get("lm_labels")
        if masked_lm_labels is None and lm_labels is None:
            # expecting huggingface style mapping
            raise Exception("batch mast have mlm_labels or lm_labels key")
        output = self.model(**batch)
        vocab_size = output[1].size(2)

        loss = output[0]
        logits = output[1].view(-1, vocab_size)
        self.batch_metrics = {"loss": loss}
        if masked_lm_labels is not None:
            self.input["targets"] = masked_lm_labels.view(-1)
            self.output = {"loss": loss, "logits": logits}
        else:
            self.input["targets"] = lm_labels.view(-1)
            self.output = {"loss": loss, "logits": logits}


texts = [
    "Bonaparte Crossing the Alps is an oil-on-canvas painting by French artist",  # noqa: E501
    "Bhaskara's Lemma is an identity used as a lemma during the chakravala method. ",  # noqa: E501
]


def test_is_running():
    """Test if perplexity is running normal"""
    if version.parse(transformers_version) >= version.parse("3.0.0"):
        return

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelWithLMHead.from_pretrained("distilbert-base-uncased")
    dataset = LanguageModelingDataset(texts, tok)
    collate_fn = DataCollatorForLanguageModeling(tok).collate_batch
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    runner = HuggingFaceRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders={"train": dataloader},
        callbacks={
            "optimizer": dl.OptimizerCallback(),
            "perplexity": PerplexityMetricCallback(),
        },
        check=True,
    )
