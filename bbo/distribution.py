import abc

import torch
from abc import ABC

from torch import Tensor


class WeightsDistributionBase(ABC):
    @abc.abstractmethod
    def distribute_weights(self, data: Tensor, quantile: int) -> Tensor:
        raise NotImplementedError()


class SigmoidWeights(WeightsDistributionBase):
    def __init__(self, gamma: int = -10):
        self.gamma = gamma

    def distribute_weights(self, data: Tensor, quantile: int) -> Tensor:
        threshold_index = int(len(data) * (1 - quantile / 100))
        kth_value, _ = torch.kthvalue(data, threshold_index)

        weights = 1 / (1 + (self.gamma * (data - kth_value)).exp())
        return weights.detach()


class QuantileWeights(WeightsDistributionBase):
    def distribute_weights(self, data: Tensor, quantile: int) -> Tensor:
        train_percentile = quantile / 100

        _, indices = torch.topk(-data, int(len(data) * train_percentile))

        weights = torch.zeros_like(data, dtype=data.dtype)
        weights[indices] = 1.0
        weights = weights.detach()

        return weights
