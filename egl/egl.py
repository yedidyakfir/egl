from typing import Callable, Type, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from .convergence import ConvergenceAlgorithm
from .distribution import WeightsDistributionBase
from .losses import loss_with_quantile
from .trainer import train_gradient_network, step_model_with_gradient
from .utils import reset_all_weights, loss_from_taylor_loss


class EGL(ConvergenceAlgorithm):
    def __init__(
        self,
        *args,
        gradient_network: Module,
        gradient_optimizer: Optimizer,
        grad_loss: Callable,
        num_of_minibatch_to_train: int,
        database_type: Type[Dataset],
        dataset_parameters: Callable,
        weight_func: WeightsDistributionBase = None,
        taylor_loss: Callable[
            [Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]
        ] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_network = gradient_network
        self.gradient_optimizer = gradient_optimizer
        self.grad_loss = grad_loss
        self.dataset_parameters = dataset_parameters
        self.num_of_minibatch_to_train = num_of_minibatch_to_train
        self.database_type = database_type
        self.weight_func = weight_func
        self.taylor_loss = taylor_loss

    def train_surrogate(self, samples: Tensor, samples_value: Tensor, batch_size: int):
        self.grad_network.train()
        mapped_evaluations = self.value_normalizer.map(samples_value)
        additional_params = self.dataset_parameters(self)
        dataset = self.database_type(
            database=samples,
            values=mapped_evaluations,
            **additional_params,
            logger=self.logger,
        )

        if self.weight_func:
            self.weight_func.pre_training(dataset.database, dataset.values)

        loss = loss_from_taylor_loss(self.taylor_loss, self.grad_loss)
        sampler = None
        if self.weight_func:
            distribute_weights = self.weight_func.distribution_from_dataset(dataset)
            if distribute_weights.device.type == "mps":
                distribute_weights = distribute_weights.cpu()
            sampler = WeightedRandomSampler(
                distribute_weights.detach().clone(), len(dataset)
            )

        train_gradient_network(
            loss,
            self.gradient_optimizer,
            dataset,
            batch_size,
            self.logger,
            # self.num_of_minibatch_to_train,
            sampler=sampler,
        )
        self.grad_network.eval()

    def calc_loss(self, value: Tensor, target: Tensor) -> Tensor:
        if self.weight_func:
            return loss_with_quantile(value, target, self.weight_func, self.grad_loss)
        return self.grad_loss(value, target)

    def train_model(self):
        self.gradient_optimizer.zero_grad()
        self.grad_network.eval()

        curr_point_gradient = self.gradient(self.curr_point)
        self.logger.info(
            f"Algorithm {self.__class__.__name__} "
            f"moving Gradient size: {torch.norm(curr_point_gradient)} on {self.env}"
        )
        step_model_with_gradient(
            self.curr_point, curr_point_gradient, self.function_opt
        )

    def gradient(self, x) -> Tensor:
        training = self.grad_network.training
        self.grad_network.eval()
        model_to_train_gradient = self.grad_network(x.unsqueeze(0)).squeeze(0)
        self.grad_network.train(training)
        model_to_train_gradient[model_to_train_gradient != model_to_train_gradient] = 0
        return model_to_train_gradient

    def explore(self, exploration_size: int):
        samples, evaluations = super().explore(exploration_size)
        if self.weight_func:
            self.weight_func.update(samples=samples, values=evaluations)
        return samples, evaluations

    def after_shrinking_hook(self):
        reset_all_weights(self.grad_network)
        if self.weight_func:
            self.weight_func.restart()
