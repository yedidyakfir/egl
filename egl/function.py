from typing import Protocol

from torch import Tensor


class Function(Protocol):
    def evaluate(self, data: Tensor) -> Tensor:
        raise NotImplementedError()

    def denormalize(self, data: Tensor) -> Tensor:
        """
        map data from (-1,1) to entire space
        """
        raise NotImplementedError()

    def normalize(self, data: Tensor) -> Tensor:
        raise NotImplementedError()
