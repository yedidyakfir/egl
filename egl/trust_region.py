import abc
from abc import ABC

import torch
from torch import Tensor

TANH_UPPER_LIMIT = 10
TANH_LOWER_LIMIT = -10


class TrustRegion(ABC):
    @abc.abstractmethod
    def map(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def squeeze(self, best_result: Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def move_center(self, best_result: Tensor):
        raise NotImplementedError()

    @abc.abstractmethod
    def unsqueeze(self):
        raise NotImplementedError()


class StaticShrinkingTrustRegion(TrustRegion, ABC):
    def __init__(
        self,
        lower_bounds: Tensor,  # NOTE: This is legacy - Should be deleted for package
        upper_bounds: Tensor,
        shrink_factor: float = 0.9,
        min_sigma: float = 0.1,
        min_trust_region_size: float = 1e-13,
        dim_proportion_factor: float = 0,
        dtype: torch.dtype = torch.float64,
        device: int = None,
    ):
        shape = lower_bounds.shape
        self.lower_bounds = lower_bounds.to(device=device)
        self.upper_bounds = upper_bounds.to(device=device)
        self.min_trust_region_size = min_trust_region_size
        self.mu = torch.zeros(shape, device=device, dtype=dtype)
        self.sigma = torch.ones(shape, device=device, dtype=dtype)
        self.dim_proportion_factor = dim_proportion_factor
        self.shrink_factor = shrink_factor
        self.min_sigma = min_sigma * torch.ones(shape, device=device, dtype=dtype)
        self.dtype = dtype

    @abc.abstractmethod
    def normalize_to_real(self, data: Tensor) -> Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def normalize_to_unreal(self, data: Tensor) -> Tensor:
        raise NotImplementedError()

    @property
    def device(self):
        return self.mu.device

    def _move_to_best_device(self, tensor: Tensor):
        tensor_device = tensor.device
        if tensor_device == torch.device("cpu"):
            tensor = tensor.to(device=self.device)
            mu, sigma = self.mu, self.sigma
        else:
            mu, sigma = (
                self.mu.to(device=tensor_device),
                self.sigma.to(device=tensor_device),
            )
        return tensor, mu, sigma

    def map(self, tensor) -> Tensor:
        original_shape = tensor.shape
        tensor, mu, sigma = self._move_to_best_device(tensor)
        tensor = (tensor - mu) / sigma.view(1, -1)
        tensor = self.normalize_to_unreal(tensor)
        return tensor.reshape(original_shape)

    def inverse(self, tensor) -> Tensor:
        tensor, mu, sigma = self._move_to_best_device(tensor)
        normalized_tensor = self.normalize_to_real(tensor)
        tensor = mu + sigma * normalized_tensor
        tensor = torch.clamp(tensor, min=-1, max=1)
        return tensor

    def squeeze(self, best_result):
        best_result = best_result.to(device=self.device)
        # max_dimensional_delta = gradient.abs().max()
        shrink_mask = (best_result < self.mu + (1 - self.min_sigma) * self.sigma) | (
            best_result > self.mu - (1 - self.min_sigma) * self.sigma
        )
        # gradient_mask = gradient.abs() > max_dimensional_delta / self.dim_proportion_factor
        if self.dim_proportion_factor:
            shrink_mask = shrink_mask  # & gradient_mask
        self.sigma[shrink_mask] = self.shrink_factor * self.sigma[shrink_mask]
        self.mu = best_result.detach()

    def move_center(self, best_result: Tensor):
        self.mu = best_result.detach()

    def unsqueeze(self):
        self.sigma = self.sigma / self.shrink_factor


class TanhTrustRegion(StaticShrinkingTrustRegion):
    def normalize_to_real(self, data: Tensor) -> Tensor:
        return torch.tanh(data)

    def normalize_to_unreal(self, data: Tensor) -> Tensor:
        data = torch.clamp(data, min=-1 + 1e-16, max=1 - 1e-16)
        return torch.arctanh(data)


class LinearTrustRegion(StaticShrinkingTrustRegion):
    def normalize_to_real(self, data: Tensor) -> Tensor:
        normalized_tensor = (
            ((data - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)) * 2
        ) - 1
        return torch.clip(normalized_tensor, min=-1, max=1)

    def normalize_to_unreal(self, data: Tensor) -> Tensor:
        return (
            0.5 * (data + 1) * (self.upper_bounds - self.lower_bounds)
            + self.lower_bounds
        )
