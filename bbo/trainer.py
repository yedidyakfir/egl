import math
from logging import Logger
from typing import List

import torch
from datasets import Dataset
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, RandomSampler, DataLoader


def train_gradient_network(
    gradient_loss_func: Module,
    optimizer,
    dataset: Dataset,
    batch_size: int,
    num_of_minibatch: int,
):
    sampler = BatchSampler(
        RandomSampler(range(num_of_minibatch)), batch_size=batch_size, drop_last=False
    )
    data_loader = DataLoader(dataset, sampler=sampler)
    for i, (x_i, x_j, y_i, y_j) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i, x_j, y_i, y_j = x_i[0], x_j[0], y_i[0], y_j[0]
        loss = gradient_loss_func(x_i, x_j, y_i, y_j)
        loss.backward()
        optimizer.step()


def step_model_with_gradient(model, gradient: Tensor, optimizer: Optimizer):
    optimizer.zero_grad()
    gradient_index = 0
    for layer in model.children():
        for model_parameter in layer.parameters():
            gradient_offset = model_parameter.numel()
            parameter_gradient = (
                gradient[gradient_index : gradient_index + gradient_offset]
                .reshape(model_parameter.shape)
                .clone()
            )
            with torch.no_grad():
                model_parameter.grad = parameter_gradient
            gradient_index += gradient_offset
    optimizer.step()
