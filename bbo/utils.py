import torch
from torch import Tensor
from torch.nn import Module


def reset_all_weights(model: Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def distance_between_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> torch.Tensor:
    return torch.norm(tensor1 - tensor2)


def ball_perturb(
    ball_center: Tensor,
    eps: float,
    num_samples: int,
    dtype: torch.dtype = torch.float64,
    device: int = None,
) -> Tensor:
    ball_dim_size = ball_center.shape[-1]

    perturb = (
        torch.FloatTensor(num_samples, ball_dim_size)
        .to(device=device, dtype=dtype)
        .normal_()
    )
    mag = torch.FloatTensor(num_samples, 1).to(device=device, dtype=dtype).uniform_()
    perturb = perturb / (torch.norm(perturb, dim=1, keepdim=True) + 1e-8)

    explore = ball_center + eps * mag * perturb
    return explore
