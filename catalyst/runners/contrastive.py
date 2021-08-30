from typing import Any

from catalyst.core.runner import IRunner
from catalyst.runners.runner import Runner
from catalyst.typing import RunnerModel


class IContrastiveRunner(IRunner):
    def __init__(
        self,
        input_key: Any = "features",
        output_key: Any = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
        encoder: RunnerModel = None,
    ):
        """Init."""
        IRunner.__init__(self)

        self._input_key = input_key
        self._output_key = output_key
        self._target_key = target_key
        self._loss_key = loss_key
        self._encoder = encoder


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
            "projection1": projection1,
            "projection2": projection2,
            "embedding1": embedding1,
            "embedding2": embedding2,
            "target": target,
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
            "projection1": projection1,
            "projection2": projection2,
            "embedding1": embedding1,
            "embedding2": embedding2,
            "target": target,
        }
