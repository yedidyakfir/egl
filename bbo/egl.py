import torch
from typing import Callable, Type

from torch.utils.data import Dataset
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from bbo.convergence import ConvergenceAlgorithm
from bbo.losses import GradientLoss
from bbo.trainer import train_gradient_network, step_model_with_gradient


class EGL(ConvergenceAlgorithm):
    def __init__(
        self,
        *args,
        gradient_network: Module,
        gradient_optimizer: Optimizer,
        grad_loss: Callable,
        num_of_minibatch_to_train: int,
        database_type: Type[Dataset],
        dataset_parameters: dict,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grad_network = gradient_network
        self.gradient_optimizer = gradient_optimizer
        self.grad_loss = grad_loss
        self.dataset_parameters = dataset_parameters
        self.num_of_minibatch_to_train = num_of_minibatch_to_train
        self.database_type = database_type

    def train_helper_model(
        self, samples: Tensor, samples_value: Tensor, batch_size: int
    ):
        self.grad_network.train()
        mapped_evaluations = self.output_mapping.map(samples_value)
        dataset = self.database_type(
            database=samples,
            values=mapped_evaluations,
            **self.dataset_parameters,
            logger=self.logger,
        )
        taylor_loss = GradientLoss(
            self.grad_network, self.perturb * self.epsilon, self.calc_loss
        )
        train_gradient_network(
            taylor_loss,
            self.gradient_optimizer,
            dataset,
            batch_size,
            self.num_of_minibatch_to_train,
        )
        self.grad_network.eval()

    def calc_loss(self, value: Tensor, target: Tensor) -> Tensor:
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
