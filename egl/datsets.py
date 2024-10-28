from logging import Logger

import torch
from torch import Tensor
from torch.utils.data import Dataset


class PointDataset(Dataset):
    def __init__(self, samples: Tensor, evaluations: Tensor):
        super().__init__()
        self.samples = samples
        self.evaluations = evaluations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.evaluations[idx]


class TuplesDataset(Dataset):
    def __init__(
        self, i_indexes: Tensor, j_indexes: Tensor, database: Tensor, values: Tensor
    ):
        super().__init__()
        self.i_indexes = i_indexes
        self.j_indexes = j_indexes
        self.database = database.detach()
        self.values = values.detach()

    def __len__(self):
        return len(self.i_indexes)

    def __getitem__(self, idx):
        j_index = self.j_indexes[idx]
        i_idx = self.i_indexes[idx]
        x_i = self.database[i_idx].detach()
        x_j = self.database[j_index].detach()
        y_i = self.values[i_idx].detach()
        y_j = self.values[j_index].detach()
        return x_i, x_j, y_i, y_j


class SinglePairPerPointDataset(TuplesDataset):
    def __init__(self, database: Tensor, values: Tensor, exploration_size: int):
        len_replay_buffer = len(values)
        i_indexes = torch.randperm(len_replay_buffer, device=database.device)

        i_reference = torch.randint(
            0, exploration_size, size=(len_replay_buffer,), device=database.device
        )
        # We are trying to batch each sample with his corresponding samples,
        # so we refer from num of exploration samples
        explore_indexes = torch.div(i_indexes, exploration_size, rounding_mode="trunc")

        j_indexes = exploration_size * explore_indexes + i_reference
        super().__init__(i_indexes, j_indexes, database, values)


class PairsInRangeDataset(TuplesDataset):
    def __init__(self, database: Tensor, values: Tensor, exploration_size: int):
        len_replay_buffer = len(values)
        single_i_indexes = torch.randperm(len_replay_buffer, device=database.device)
        i_indexes = single_i_indexes.repeat(exploration_size)

        i_reference_single = torch.randint(
            0, exploration_size, size=(len_replay_buffer,), device=database.device
        )
        i_reference = torch.cat(
            [
                (i_reference_single + i) % exploration_size
                for i in range(exploration_size)
            ],
            axis=0,
        )
        # We are trying to batch each sample with his corresponding samples,
        # so we refer from num of exploration samples
        explore_indexes = torch.div(i_indexes, exploration_size, rounding_mode="trunc")

        j_indexes = exploration_size * explore_indexes + i_reference
        non_same_samples = i_indexes != j_indexes
        super().__init__(
            i_indexes[non_same_samples], j_indexes[non_same_samples], database, values
        )


class PairsInEpsRangeDataset(TuplesDataset):
    def __init__(
        self,
        database: Tensor,
        values: Tensor,
        epsilon: float,
        max_tuples: int = None,
        logger: Logger = None,
    ):
        database = database.detach()
        distances = torch.cdist(database, database)

        only_upper_half_matrix = torch.triu(
            torch.ones_like(distances, device=database.device, dtype=torch.bool),
            diagonal=1,
        )
        close_to_epsilon = distances < epsilon
        distances[~(close_to_epsilon & only_upper_half_matrix)] = 0
        tuples = distances.nonzero()
        self.size = max_tuples if max_tuples else len(tuples)
        if len(tuples) == 0:
            logger.error(
                f"No tuples, min distance: {distances.min()}, max distance: {distances.max()} epsilon {epsilon}"
            )
            super().__init__(
                torch.tensor([], device=database.device),
                torch.tensor([], device=database.device),
                database,
                values,
            )
        else:
            tuples_idx = torch.randint(len(tuples), (self.size,))
            super().__init__(
                tuples[tuples_idx, 0], tuples[tuples_idx, 1], database, values
            )

    def __len__(self):
        return self.size


class NewPairEpsDataset(Dataset):
    def __init__(
        self,
        database: Tensor,
        values: Tensor,
        epsilon: float,
        new_samples: Tensor,
    ):
        super().__init__()
        self.database = database.detach()
        self.new_samples = new_samples
        self.values = values
        distances = torch.cdist(new_samples, database)

        only_upper_half_matrix = torch.triu(
            torch.ones_like(distances, device=database.device, dtype=torch.bool),
            diagonal=1,
        )
        close_to_epsilon = distances < epsilon
        distances[~(close_to_epsilon & only_upper_half_matrix)] = 0
        self.tuples = distances.nonzero()

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        j_index = self.tuples[idx, 0]
        i_idx = self.tuples[idx, 1]
        x_i = self.database[i_idx].detach()
        x_j = self.new_samples[j_index].detach()
        y_i = self.values[i_idx].detach()
        y_j = self.values[j_index].detach()
        return x_i, x_j, y_i, y_j


class PairFromDistributionDataset(TuplesDataset):
    def __init__(self, dataset: TuplesDataset):
        values = dataset.values.detach()
        values_diff = (values[dataset.i_indexes] - values[dataset.j_indexes]).abs()
        values_diff_distributions = values_diff / values_diff.sum()
        distributed_indices = torch.multinomial(
            values_diff_distributions, len(values_diff)
        )
        super().__init__(
            i_indexes=dataset.i_indexes[distributed_indices],
            j_indexes=dataset.j_indexes[distributed_indices],
            database=dataset.database,
            values=dataset.values,
        )
