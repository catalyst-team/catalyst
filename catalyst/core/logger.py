from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import numpy as np


# origin:
# https://github.com/lanpa/tensorboardX/blob/master/tensorboardX/writer.py
class ILogger(ABC):
    @property
    def name(self) -> str:
        pass

    @property
    def logdir(self) -> str:
        pass

    @abstractmethod
    def log_scalar(
        self, value: float, name: str, step: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None,
    ) -> None:
        pass

    # not sure if
    # def log_histogram(
    #     self, histogram: np.ndarray, step: Optional[int] = None,
    # ) -> None:
    #     pass

    def log_image(
        self, image: np.ndarray, step: Optional[int] = None,
    ) -> None:
        pass

    def log_graph(self, model: Any) -> None:
        pass

    @abstractmethod
    def log_hparams(self, hparams: Dict) -> None:
        pass
