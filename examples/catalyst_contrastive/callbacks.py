from typing import Dict, Tuple
from pathlib import Path
import pickle

import torch

from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.metrics._metric import ICallbackLoaderMetric


class FeatureAccumulator(ICallbackLoaderMetric):
    """Feature bank,
    Args:
        prefix: embeddings accumulator prefix
        suffix: embeddings accumulator suffix
    """

    def __init__(self, save_path, prefix: str = None, suffix: str = None):
        """Init."""
        super().__init__(compute_on_call=False, prefix=prefix, suffix=suffix)
        self.accaulator_name = f"{self.prefix}feature_accumulator{self.suffix}"
        self.features = []
        self.targets = []
        self.save_path = save_path

    def reset(self, loader_name, epoch) -> None:
        """Resets all fields"""
        self.features = []
        self.targets = []
        self.loader_name = loader_name
        self.epoch = epoch

    def update(self, features: torch.tensor, targets: torch.Tensor) -> None:
        """Updates accumulator with features and targets for new data.
        Args:
            features: tensor with features
            targets: tensor with targets
        """
        self.features.append(features.cpu().detach())
        self.targets.append(targets.cpu().detach())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Concat computed features and targets."""
        targets = torch.cat(self.targets)
        features = torch.cat(self.features)
        return features, targets

    def compute_key_value(self) -> Dict[str, float]:
        """Save computed features with targets."""
        features, targets = self.compute()

        feature_bank = {"feature": features, "targets": targets}

        file_name = f"{self.loader_name}_{self.epoch}_embs.pkl"
        file_name = Path(self.save_path) / file_name
        with open(file_name, "wb") as handle:
            pickle.dump(feature_bank, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return {}


class FeatureAccumulatorCallback(LoaderMetricCallback):
    """Feature accumulator callback.
    Args:
        input_key: input key to use for features, specifies our ``features``.
        target_key: output key to use for targets, specifies our ``y_true``.
        prefix: accumulator prefix
        suffix: accumulator suffix
    """

    def __init__(
        self, save_path, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=FeatureAccumulator(save_path, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key,
        )

    def on_loader_start(self, runner) -> None:
        """On loader star action: reset metric values in case of ICallbackLoaderMetric metric

        Args:
            runner: current runner
        """
        self.metric.reset(loader_name=runner.loader_key, epoch=runner.global_epoch_step)
