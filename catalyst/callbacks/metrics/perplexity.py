from functools import partial

from torch import nn

from catalyst.callbacks.metric import BatchMetricCallback


def _perplexity_metric(outputs, targets, criterion):
    cross_entropy = criterion(outputs, targets).detach()
    perplexity = 2 ** cross_entropy
    return perplexity


class PerplexityCallback(BatchMetricCallback):
    """
    Perplexity is a very popular metric in NLP
    especially in Language Modeling task.
    It is 2^cross_entropy.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "perplexity",
        ignore_index: int = None,
        **kwargs,
    ):
        """
        Args:
            input_key: input key to use for perplexity calculation,
                target tokens
            output_key: output key to use for perplexity calculation,
                logits of the predicted tokens
            ignore_index: index to ignore, usually pad_index
        """
        self.ignore_index = ignore_index or nn.CrossEntropyLoss().ignore_index
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        metric_fn = partial(_perplexity_metric, criterion=self.cross_entropy_loss)

        super().__init__(
            metric_fn=metric_fn,
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
            **kwargs,
        )


__all__ = ["PerplexityCallback"]
