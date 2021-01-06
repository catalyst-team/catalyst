from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np


# origin:
# https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/writer.py
class ILogger(ABC):
    """Interface for all Loggers."""

    @property
    def name(self) -> str:
        """Returns corresponding logger name.

        Returns:
            name of the associated logger.  # noqa: DAR202

        Raises:
            NotImplementedError: if name was not specified.
        """
        raise NotImplementedError()

    @property
    def logdir(self) -> str:
        """Returns corresponding logdir path.

        Returns:
            path to associated logdir.  # noqa: DAR202

        Raises:
            NotImplementedError: if logdir was not specified.
        """
        raise NotImplementedError()

    @abstractmethod
    def log_scalar(
        self, value: float, name: str, step: Optional[int] = None,
    ) -> None:
        """Logs scalar.

        Args:
            value: scalar value.
            name: scalar name.
            step: experiment step.
        """
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None,
    ) -> None:
        """Logs metrics.

        Args:
            metrics: key-value storage with metrics.
            step: experiment step.
        """
        pass

    @abstractmethod
    def log_histogram(
        self, histogram: np.ndarray, step: Optional[int] = None,
    ) -> None:
        """Logs histogram.

        Args:
            histogram: numpy array with histogram, shape - ??
            step: experiment step
        """
        pass

    @abstractmethod
    def log_image(
        self, image: np.ndarray, step: Optional[int] = None,
    ) -> None:
        """Logs image.

        Args:
            image: numpy array with image, [h; w; 3/1]
            step: experiment step
        """
        pass

    @abstractmethod
    def log_graph(self, model: Any) -> None:
        """Logs experiment model.

        Args:
            model: some model, for PyTorch – `nn.Module`.
        """
        pass

    @abstractmethod
    def log_hparams(self, hparams: Dict) -> None:
        """Logs experiment hyperparameters.

        Args:
            hparams: key-value experiment hyperparameters.
        """
        pass
