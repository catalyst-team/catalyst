from torch import nn

from catalyst.core.callbacks import BatchMetricCallback


class PerplexityMetricCallback(BatchMetricCallback):
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
    ):
        """
        Args:
            input_key (str): input key to use for perplexity calculation,
                target tokens
            output_key (str): output key to use for perplexity calculation,
                logits of the predicted tokens
            ignore_index (int): index to ignore, usually pad_index
        """
        self.ignore_index = ignore_index or nn.CrossEntropyLoss().ignore_index
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index
        )
        super().__init__(
            metric_fn=self.metric_fn,
            input_key=input_key,
            output_key=output_key,
            prefix=prefix,
        )

    def metric_fn(self, outputs, targets):
        """Calculate perplexity

        Args:
            outputs: model output
            targets: model targets

        Returns:
            computed perplexity metric
        """
        cross_entropy = (
            self.cross_entropy_loss(outputs, targets).detach().cpu()
        )
        perplexity = 2 ** cross_entropy
        return perplexity.item()
