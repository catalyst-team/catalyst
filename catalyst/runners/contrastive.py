from typing import Any

from catalyst.core.runner import IRunner
from catalyst.runners.runner import Runner
from catalyst.typing import RunnerModel


class IContrastiveRunner(IRunner):
    """IRunner for experiments with contrastive model.

    Args:
        input_key: key in ``runner.batch`` dict mapping for model input
        output_key: key for ``runner.batch`` to store model output
        target_key: key in ``runner.batch`` dict mapping for target
        loss_key: key for ``runner.batch_metrics`` to store criterion loss output
        projection_key: key for ``runner.batch`` to store model projection
        embedding_key: key for `runner.batch`` to store model embeddings

    Abstraction, please check out implementations for more details:

        - :py:mod:`catalyst.runners.contrastive.ContrastiveRunner`

    .. note::
        IContrastiveRunner contains only the logic with batch handling.


    ISupervisedRunner logic pseudocode:

    .. code-block:: python

        batch = {...}

    .. note::
        Please follow the `minimal examples`_ sections for use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples

    """

    def __init__(
        self,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
        projection_key: str = "projections",
        embedding_key: str = "embeddings",
    ):
        """Init."""
        IRunner.__init__(self)

        self._input_key = input_key
        self._output_key = output_key
        self._target_key = target_key
        self._loss_key = loss_key
        self._projection_key = projection_key
        self._embedding_key = embedding_key


class ContrastiveRunner(IContrastiveRunner, Runner):
    def predict_batch(self, batch):
        # model train/valid step
        # unpack the batch
        sample_aug1, sample_aug2, target = batch
        embedding1 = self._encoder(sample_aug1)
        embedding2 = self._encoder(sample_aug2)
        projection1 = self.model(embedding1)
        projection2 = self.model(embedding2)
        self.batch = {
            f"{self._projection_key}_1": projection1,
            f"{self._projection_key}_2": projection2,
            f"{self._embedding_key}_1": embedding1,
            f"{self._embedding_key}_2": embedding2,
            self._target_key: target,
        }

    def handle_batch(self, batch):
        # model train/valid step
        # unpack the batch
        sample_aug1, sample_aug2, target = batch
        embedding1 = self._encoder(sample_aug1)
        embedding2 = self._encoder(sample_aug2)
        projection1 = self.model(embedding1)
        projection2 = self.model(embedding2)
        self.batch = {
            f"{self._projection_key}_1": projection1,
            f"{self._projection_key}_2": projection2,
            f"{self._embedding_key}_1": embedding1,
            f"{self._embedding_key}_2": embedding2,
            self._target_key: target,
        }

    def train(self, encoder: RunnerModel, *args, **kwargs) -> None:
        self._encoder = encoder
        super().train(self, *args, **kwargs)
