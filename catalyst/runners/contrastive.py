from typing import Any, Mapping

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
        augemention_key: str = "aug",
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
        self._augemention_key = augemention_key
        self._embedding_key = embedding_key

    def _process_batch(self, batch):
        if isinstance(batch, (tuple, list)):
            assert len(batch) == 3
            batch = {
                f"{self._augemention_key}_1": batch[0],
                f"{self._augemention_key}_2": batch[1],
                self._target_key: batch[2],
            }
        return batch

    def _process_input(self, batch: Mapping[str, Any], **kwargs):
        embedding1, projection1 = self.model(batch[f"{self._augemention_key}_1"], **kwargs)
        embedding2, projection2 = self.model(batch[f"{self._augemention_key}_2"], **kwargs)
        
        batch = {
            **batch,
            f"{self._projection_key}_1": projection1,
            f"{self._projection_key}_2": projection2,
            f"{self._embedding_key}_1": embedding1,
            f"{self._embedding_key}_2": embedding2,
        }
        return batch
    
    def forward(self, batch: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
        """
        Forward method for your Runner.
        Should not be called directly outside of runner.
        If your model has specific interface, override this method to use it

        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoaders.
            **kwargs: additional parameters to pass to the model

        Returns:
            dict with model output batch
        """
        return self._process_input(batch, **kwargs)
    


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
