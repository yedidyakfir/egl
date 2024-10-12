import abc

import torch
from abc import ABC

from torch import Tensor


class WeightsDistributionBase(ABC):
    @abc.abstractmethod
    def distribute_weights(self, data: Tensor) -> Tensor:
        raise NotImplementedError()


class SigmoidWeights(WeightsDistributionBase):
    def __init__(self, gamma: int = -10, quantile: int = 83):
        self.gamma = gamma
        self.quantile = quantile

    def distribute_weights(self, data: Tensor) -> Tensor:
        threshold_index = int(len(data) * (1 - self.quantile / 100))
        kth_value, _ = torch.kthvalue(data, threshold_index)

        weights = 1 / (1 + (self.gamma * (data - kth_value)).exp())
        return weights.detach()


class QuantileWeights(WeightsDistributionBase):
    def __init__(self, quantile: int = 83):
        self.quantile = quantile

    def distribute_weights(self, data: Tensor) -> Tensor:
        train_percentile = self.quantile / 100

        _, indices = torch.topk(-data, int(len(data) * train_percentile))

        weights = torch.zeros_like(data, dtype=data.dtype)
        weights[indices] = 1.0
        weights = weights.detach()

        return weights
