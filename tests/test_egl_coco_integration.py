import pytest
import torch
from cocoex import Suite

from egl import minimize


def create_coco_function(func_num: int, dim: int, instance: int):
    suite = Suite("bbob", "", "")
    problem = suite.get_problem_by_function_dimension_instance(func_num, dim, instance)

    def coco_callable(x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        results = [problem(xi) for xi in x_np]
        return torch.tensor(results, dtype=x.dtype, device=x.device)

    return coco_callable, problem.lower_bounds[0], problem.upper_bounds[0]


@pytest.mark.parametrize(
    ["func_num", "dim", "instance"],
    [
        [12, 20, 2],  # Sphere
        # [1, 5, 1],  # Sphere
        # [2, 5, 1],  # Ellipsoid separable
    ],
)
def test_egl_with_coco_runs_without_error_sanity(func_num, dim, instance):
    # Arrange
    import logging

    logging.basicConfig(level=logging.INFO)
    coco_func, lower, upper = create_coco_function(func_num, dim, instance)
    x0 = torch.zeros(dim, dtype=torch.float64)

    # Act
    result = minimize(
        fun=coco_func,
        x0=x0,
        args=[False, True],
        bounds=[lower, upper],
    )

    # Assert
    assert result is not None
    assert result.shape == (dim,)
