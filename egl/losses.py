from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from .distribution import WeightsDistributionBase
from .utils import hessian_from_gradient_network


class GradientLoss(Module):
    def __init__(self, grad_network):
        super().__init__()
        self.grad_network = grad_network

    def forward(self, x_i, x_j, y_i, y_j):
        assert len(x_i.shape) <= 2, "Cant handle multiple batches yet"
        grad_x_i = self.grad_network(x_i)

        x_delta = x_j - x_i
        value = (x_delta * grad_x_i).sum(dim=1) + self.taylor_remainder(x_delta)
        target = y_j - y_i

        return value, target

    def taylor_remainder(self, x_delta):
        return 0


class NaturalHessianLoss(GradientLoss):
    def calculate_hessian(self, x):
        return hessian_from_gradient_network(self.grad_network, x)

    def taylor_remainder(self, x_delta):
        hessian = self.calculate_hessian(x_delta)
        return torch.bmm(
            torch.bmm(
                x_delta.reshape((x_delta.shape[0], 1, x_delta.shape[1])), hessian
            ),
            x_delta.reshape((x_delta.shape[0], x_delta.shape[1], 1)),
        ).squeeze()


class DetachedHessianLoss(NaturalHessianLoss):
    def calculate_hessian(self, x):
        with torch.no_grad():
            return super().calculate_hessian(x).detach()


def loss_with_quantile(
    value: Tensor,
    target: Tensor,
    weights_creator: WeightsDistributionBase,
    loss: Callable,
) -> Tensor:
    smallest_element = (value - target).abs().clone().detach()
    weights = weights_creator.distribute_weights(smallest_element)

    loss = loss(value, target)
    loss = (loss * weights).mean()
    return loss
