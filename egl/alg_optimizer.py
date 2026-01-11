import torch
import math

from torch.optim import Adam

from .datsets import PairsInEpsRangeDataset
from .distribution import QuantileWeights
from .egl import EGL
from .function import BasicFunction
import torch.nn as nn

from .handlers import CallableForEpochEnd
from .losses import GradientLoss, NaturalHessianLoss
from .trust_region import TanhTrustRegion
from .value_normalizer import AdaptedOutputUnconstrainedMapping


def minimize(fun, x0, args=(), bounds=None, callback=None):
    use_hessian = args[0]
    taylor_loss_klass = NaturalHessianLoss if use_hessian else GradientLoss
    use_weights = args[1]
    dim = len(x0)
    x0.requires_grad = True
    bounds = bounds or [-5, 5]
    func = BasicFunction(fun, *bounds)
    optimizer = Adam(x0.parameters(), lr=0.1)
    gradient_network = nn.Sequential(
        nn.Linear(dim, 10),
        nn.ReLU(),
        nn.Linear(10, 15),
        nn.ReLU(),
        nn.Linear(15, dim),
    ).to(device=x0.device, dtype=x0.dtype)
    grad_optimizer = Adam(gradient_network.parameters())
    taylor_loss = taylor_loss_klass(gradient_network)
    num_of_minibatch_to_train = int(2000 * math.sqrt(dim))
    egl = EGL(
        env=func,
        curr_point=x0,
        function_opt=optimizer,
        epsilon=0.4 * math.sqrt(dim),
        epsilon_factor=0.97,
        min_epsilon=1e-4,
        perturb=0,
        gradient_network=gradient_network,
        gradient_optimizer=grad_optimizer,
        grad_loss=nn.SmoothL1Loss(),
        num_of_minibatch_to_train=num_of_minibatch_to_train,
        database_type=PairsInEpsRangeDataset,
        dataset_parameters=lambda alg: {
            "epsilon": alg.epsilon,
            "max_tuples": num_of_minibatch_to_train,
        },
        weight_func=QuantileWeights() if use_weights else None,
        taylor_loss=taylor_loss,
        trust_region=TanhTrustRegion(
            lower_bounds=torch.tensor(
                [-func.lower_bound] * dim, device=x0.device, dtype=x0.dtype
            ),
            upper_bounds=torch.tensor(
                [func.lower_bound] * dim, device=x0.device, dtype=x0.dtype
            ),
        ),
        value_normalizer=AdaptedOutputUnconstrainedMapping(),
    )

    egl.train(
        epochs=50_000,
        exploration_size=int(8 * math.sqrt(dim)),
        num_loop_without_improvement=10,
        min_iteration_before_shrink=40,
        callback_handlers=[CallableForEpochEnd(callback)],
    )
    return egl.env.denormalize(egl.trust_region.inverse(egl.best_model))
