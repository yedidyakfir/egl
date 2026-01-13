import abc
from abc import ABC

import numpy as np
import torch
from cma import CMAEvolutionStrategy
from egl.datsets import TuplesDataset
from torch import Tensor
from torch.distributions import MultivariateNormal


def prepare_batches_calc(dataset: TuplesDataset, max_memory_mb: int):
    # Batched implementation to respect memory constraints
    total_size = len(dataset.i_indexes)
    dim_size = dataset.database.shape[1]
    device = dataset.database.device
    dtype = dataset.database.dtype

    # Estimate memory per sample (rough approximation)
    # Each sample requires x_i, x_j (2 * dim_size floats)
    estimated_mem_per_sample = (
        2 * dim_size * torch.tensor([], dtype=dtype).element_size()
    )

    # Calculate batch size based on max_memory
    max_mem_bytes = max_memory_mb * (1024**2)
    batch_size = max(1, int(max_mem_bytes / estimated_mem_per_sample))

    # Initialize empty weights tensor
    all_weights = torch.zeros(total_size, device=device, dtype=dtype)
    return all_weights, batch_size, total_size


class WeightsDistributionBase(ABC):
    def __init__(self, max_memory_usage: int = None):
        self.max_memory_usage = max_memory_usage

    def update(self, *args, **kwargs):
        pass

    def restart(self):
        pass

    def pre_training(self, samples: Tensor, values: Tensor):
        pass

    @abc.abstractmethod
    def distribute_weights(
        self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor
    ) -> Tensor:
        raise NotImplementedError()

    def _weights_for_batch(self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor):
        return self.distribute_weights(x_i, x_j, y_i, y_j)

    def _post_process_weights(self, weights: Tensor):
        return weights

    def distribution_from_dataset(self, dataset: TuplesDataset):
        if self.max_memory_usage is None:
            # Original implementation - process all at once
            x_i = dataset.database[dataset.i_indexes]
            x_j = dataset.database[dataset.j_indexes]
            y_i = dataset.values[dataset.i_indexes]
            y_j = dataset.values[dataset.j_indexes]
            weights = self.distribute_weights(x_i, x_j, y_i, y_j).detach().clone()
            return weights
        else:
            all_weights, batch_size, total_size = prepare_batches_calc(
                dataset, self.max_memory_usage
            )

            # Process in batches
            for start_idx in range(0, total_size, batch_size):
                end_idx = min(start_idx + batch_size, total_size)

                # Get batch indices
                batch_i_indexes = dataset.i_indexes[start_idx:end_idx]
                batch_j_indexes = dataset.j_indexes[start_idx:end_idx]

                # Extract batch data
                x_i = dataset.database[batch_i_indexes]
                x_j = dataset.database[batch_j_indexes]
                y_i = dataset.values[batch_i_indexes]
                y_j = dataset.values[batch_j_indexes]

                # Process batch
                batch_weights = self._weights_for_batch(x_i, x_j, y_i, y_j).detach()

                # Store results
                all_weights[start_idx:end_idx] = batch_weights

            final_weights = self._post_process_weights(all_weights)
            return final_weights


class QuantileWeights(WeightsDistributionBase):
    def __init__(self, train_quantile: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_quantile = train_quantile
        self.quantile_value = None

    def pre_training(self, samples: Tensor, values: Tensor):
        self.quantile_value = torch.quantile(values, self.train_quantile / 100)

    def distribute_weights(
        self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor
    ) -> Tensor:
        data = torch.min(y_i, y_j)
        indices = data < self.quantile_value
        weights = torch.zeros_like(data, dtype=data.dtype)
        weights[indices] = 1.0
        return weights.detach()


class CMAWeights(WeightsDistributionBase):
    def __init__(self, dims: int, device: int, dtype: torch.dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dims = dims
        self.device = device
        self.dtype = dtype
        self.cma = None
        self.mvn = None
        self.sum_probs = None
        self.database_size = None
        self.restart()

    def update(self, *args, samples: Tensor, values: Tensor, **kwargs):
        self.cma.ask()
        self.cma.tell(samples.detach().cpu().numpy(), values.detach().cpu().numpy())
        self.mvn_from_cma()

    def restart(self):
        self.cma = CMAEvolutionStrategy(np.array([0.0] * self.dims), 0.5)
        self.mvn_from_cma()

    def mvn_from_cma(self):
        covariance_matrix = torch.from_numpy(self.cma.sm.covariance_matrix).to(
            device=self.device, dtype=self.dtype
        )
        # cov_sym = (covariance_matrix + covariance_matrix.T) / 2
        cov_sym = covariance_matrix
        w, v = torch.linalg.eigh(cov_sym)
        w_clipped = w.clamp(min=1e-6)
        c_psd = v @ w_clipped.diag() @ v.T
        c_psd = (c_psd + c_psd.T) / 2

        mean = torch.from_numpy(self.cma.mean).to(device=self.device, dtype=self.dtype)
        self.mvn = MultivariateNormal(mean, c_psd)

    def pre_training(self, samples: Tensor, values: Tensor):
        log_probs = self.mvn.log_prob(samples)
        probs = torch.exp(log_probs)
        probs[probs.isnan()] = 0
        self.sum_probs = probs.sum()
        self.database_size = len(samples)

    def distribute_weights(
        self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor
    ) -> Tensor:
        points = torch.where((y_i < y_j).unsqueeze(1), x_i, x_j).detach()
        log_probs = self.mvn.log_prob(points)
        probs = torch.exp(log_probs)
        probs[probs.isnan()] = 0
        multiplier = self.database_size // 10
        probability = probs * multiplier / (self.sum_probs.clip(min=1e-17))
        return probability.detach()


class CMAWeightsPerSample(CMAWeights):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = None
        self.values = None

    def pre_training(self, samples: Tensor, values: Tensor):
        self.samples = samples
        self.values = values
        self.restart()

    def distribute_weights(
        self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor
    ) -> Tensor:
        a_exp = self.samples.unsqueeze(0)
        b_exp = x_i.unsqueeze(1)
        matches = (a_exp == b_exp).all(dim=2)
        indexes = torch.where(
            matches, torch.arange(a_exp.size(0)).to(device=matches.device), -1
        )
        indexes = torch.max(indexes, dim=1).values
        x_i_values = y_i[indexes]
        self.update(samples=x_i, values=x_i_values)

        b_exp = x_i.unsqueeze(1)
        matches = (a_exp == b_exp).all(dim=2)
        indexes = torch.where(
            matches, torch.arange(a_exp.size(0)).to(device=matches.device), -1
        )
        indexes = torch.max(indexes, dim=1).values
        x_j_values = y_j[indexes]
        self.update(samples=x_j, values=x_j_values)
        self.mvn_from_cma()
        super().pre_training(samples=x_i, values=y_i)
        return super().distribute_weights(x_i, x_j, y_i, y_j)


class WeightDistributionNormalization(WeightsDistributionBase):
    def __init__(self, weights_creator: WeightsDistributionBase, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_creator = weights_creator

    def distribute_weights(
        self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor
    ) -> Tensor:
        weights = self.weights_creator.distribute_weights(x_i, x_j, y_i, y_j)
        weights_sum = weights.sum()
        if weights_sum == 0:
            weights = torch.ones_like(weights)
            weights_sum = weights.sum()
        return (weights / weights_sum).detach()


class WeightSmoother(WeightsDistributionBase):
    def __init__(
        self,
        inner_weights_creator: WeightsDistributionBase,
        smooth_alpha: float = 0.4,
        smooth_epsilon: float = 1e-4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.inner_weights_creator = inner_weights_creator
        self.smooth_alpha = smooth_alpha
        self.smooth_epsilon = smooth_epsilon

    def update(self, *args, **kwargs):
        self.inner_weights_creator.update(*args, **kwargs)

    def restart(self):
        self.inner_weights_creator.restart()

    def pre_training(self, samples: Tensor, values: Tensor):
        self.inner_weights_creator.pre_training(samples=samples, values=values)

    def distribute_weights(
        self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor
    ) -> Tensor:
        weights = self.inner_weights_creator.distribute_weights(x_i, x_j, y_i, y_j)
        normalized_weights = (weights + self.smooth_epsilon).pow(self.smooth_alpha)
        final_weights = normalized_weights / normalized_weights.sum()
        final_weights[final_weights.isnan()] = 0
        return final_weights.detach()

    def _post_process_weights(self, weights: Tensor):
        normalized_weights = (weights + self.smooth_epsilon).pow(self.smooth_alpha)
        final_weights = normalized_weights / normalized_weights.sum()
        final_weights[final_weights.isnan()] = 0
        return final_weights

    def _weights_for_batch(self, x_i: Tensor, x_j: Tensor, y_i: Tensor, y_j: Tensor):
        weights = self.inner_weights_creator.distribute_weights(x_i, x_j, y_i, y_j)
        return weights.detach()
