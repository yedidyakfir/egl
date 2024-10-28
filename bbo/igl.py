from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from .convergence import ConvergenceAlgorithm
from .datsets import PointDataset


class IGL(ConvergenceAlgorithm):
    def __init__(
        self,
        *args,
        training_epochs: int,
        surrogate_model: Module,
        surrogate_opt: Optimizer,
        loss: Callable,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.training_epochs = training_epochs
        self.surrogate_model = surrogate_model
        self.surrogate_opt = surrogate_opt
        self.loss = loss

    def train_surrogate(self, samples: Tensor, samples_value: Tensor, batch_size: int):
        self.surrogate_model.train()
        mapped_evaluations = self.value_normalizer.map(samples_value)
        dataset = PointDataset(samples, mapped_evaluations)
        for _ in range(self.training_epochs):
            num_of_minibatch = len(samples) // batch_size
            sampler = BatchSampler(
                RandomSampler(range(num_of_minibatch)),
                batch_size=batch_size,
                drop_last=False,
            )
            data_loader = DataLoader(dataset, sampler=sampler)
            for i_index in data_loader:
                x_i = samples[i_index]
                y_i = mapped_evaluations[i_index]

                self.surrogate_opt.zero_grad()
                self.function_opt.zero_grad()
                predicted_value = self.surrogate_model(x_i)

                loss = self.loss(predicted_value.squeeze(), y_i)
                loss.backward()
                self.surrogate_opt.step()
        self.surrogate_model.eval()

    def train_model(self):
        self.function_opt.zero_grad()
        loss = self.surrogate_model(self.curr_point)
        loss.backward()
        self.function_opt.step()
        self.logger.info(
            f"Algorithm {self.__class__.__name__} updated after loss {loss}"
        )
