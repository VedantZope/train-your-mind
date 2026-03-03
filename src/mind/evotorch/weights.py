from __future__ import annotations

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    return parameters_to_vector([p for p in model.parameters() if p.requires_grad])


def set_params_from_vector(model: torch.nn.Module, vector: torch.Tensor) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    vector_to_parameters(vector, params)
