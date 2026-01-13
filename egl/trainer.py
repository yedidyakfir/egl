from logging import Logger
from typing import List, Tuple

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, RandomSampler, DataLoader, Sampler

from .datsets import TuplesDataset


def train_gradient_network(
    gradient_loss_func: Module,
    optimizer,
    dataset: TuplesDataset,
    batch_size: int,
    logger: Logger,
    sampler: Sampler = None,
) -> Tuple[List[float], List[float]]:
    losses = []

    sampler = sampler or RandomSampler(range(len(dataset)))
    sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    data_loader = DataLoader(dataset, sampler=sampler)
    for i, (x_i, x_j, y_i, y_j) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i, x_j, y_i, y_j = x_i[0], x_j[0], y_i[0], y_j[0]
        loss = gradient_loss_func(x_i, x_j, y_i, y_j)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
    return losses


def step_model_with_gradient(point: Tensor, gradient: Tensor, optimizer: Optimizer):
    optimizer.zero_grad()
    point.grad = gradient
    optimizer.step()
