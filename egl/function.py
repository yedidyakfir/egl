from typing import Protocol

from torch import Tensor

from .exception import NoMoreBudgetError


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


class BasicFunction(Function):
    def __init__(self, func, budget=150_000, lower_bound=-5, upper_bound=5):
        self.func = func
        self.curr_budget = 0
        self.budget = budget
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, data: Tensor) -> Tensor:
        self.curr_budget += len(data)
        if self.curr_budget > self.budget:
            raise NoMoreBudgetError(f"Exceeded budget of {self.curr_budget}")
        if data.ndim == 1:
            data = data.unsqueeze(0)
            return self.func(data).squeeze(0)
        return self.func(data)

    def denormalize(self, data: Tensor) -> Tensor:
        return (
            0.5 * (data + 1) * (self.upper_bound - self.lower_bound) + self.lower_bound
        )

    def normalize(self, data: Tensor) -> Tensor:
        return (
            (data - self.lower_bound) / (self.upper_bound - self.lower_bound) * 2
        ) - 1
