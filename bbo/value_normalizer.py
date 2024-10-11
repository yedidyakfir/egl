import torch
from abc import ABC

from torch import Tensor


class ValueNormalizer(ABC):
    def map(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError()

    def inverse(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError()

    def adapt(self, new_data: Tensor):
        pass


class AdaptedOutputUnconstrainedMapping(ValueNormalizer):
    def __init__(
        self,
        outlier: float = 0.1,
        norm_lr: float = 0.1,
        output_epsilon: float = 1e-5,
        squash_eps: float = 1e-5,
        y1: float = -1,
        y2: float = 1,
    ):
        """
        This class is mapping using 2 functions
        1.  linear function that with these points (Q[outlier], y1), (Q[1-outlier], y2)
            Q[outlier] is the outlier percentile of the data
        2.  log function (depends on the output of the first function)
        This class also adapt the linear mapping slowly and give more gravity for new data
        :param epsilon: infinitesimal number to avoid dividing by zero
        """
        self.outlier = outlier
        self.output_epsilon = output_epsilon
        self.norm_lr = norm_lr
        self.squash_eps = squash_eps
        self.y1 = y1
        self.y2 = y2
        self.m = None
        self.n = None

    def adapt(self, new_data):
        data_size = len(new_data)
        outlier = max(int(data_size * self.outlier), 1)

        x2, _ = torch.kthvalue(new_data, data_size - outlier, dim=0)
        x1, _ = torch.kthvalue(new_data, outlier, dim=0)

        m = (self.y2 - self.y1) / (x2 - x1 + self.output_epsilon)
        n = self.y2 - m * x2

        if self.m is None or self.n is None:
            self.m = m
            self.n = n
        else:
            self.m = (1 - self.norm_lr) * self.m + self.norm_lr * m
            self.n = (1 - self.norm_lr) * self.n + self.norm_lr * n

    def map(self, tensor):
        if not self.m or not self.n:
            return tensor
        tensor = tensor.to(device=self.m.device) * self.m + self.n

        x_clamp_up = torch.clamp(tensor, min=self.squash_eps)
        x_clamp_down = torch.clamp(tensor, max=-self.squash_eps)

        x_log_up = torch.log(x_clamp_up) + 1
        x_log_down = -torch.log(-x_clamp_down) - 1
        tensor = (
            x_log_up * (tensor >= 1).float()
            + tensor * (tensor >= -1).float() * (tensor < 1).float()
            + x_log_down * (tensor < -1).float()
        )
        return tensor

    def inverse(self, tensor):
        x_clamp_up = torch.clamp(tensor, min=self.squash_eps)
        x_clamp_down = torch.clamp(tensor, max=-self.squash_eps)

        x_exp_up = torch.exp(x_clamp_up - 1)
        x_exp_down = -torch.exp(-(x_clamp_down + 1))

        tensor = (
            x_exp_up * (tensor >= 1).float()
            + tensor * (tensor >= -1).float() * (tensor < 1).float()
            + x_exp_down * (tensor < -1).float()
        )

        tensor = (tensor - self.n) / self.m
        return tensor
