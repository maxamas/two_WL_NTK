from layers import get_two_wl_aggregation_layer
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union
from neural_tangents import Kernel
from neural_tangents._src.stax.requirements import (
    Bool,
    Diagonal,
    get_diagonal_outer_prods,
    layer,
    mean_and_var,
    requires,
    supports_masking,
)
from jax import numpy as np
import jax
from neural_tangents import stax


def get_network_configuration(dataset, method, configuration):
    """
    dataset: str
    method: gcn or twl
    configuration: ntk configuration (initalization)
    of the network or dg for gradient descent initalization.
    """

    if dataset == "MUTAG" and method == "twl" and configuration == "gd":

        parameterization = "standard"
        n_nodes = 28
        layer_wide = 32
        two_wl_aggregation_layer = get_two_wl_aggregation_layer(
            parameterization, n_nodes, layer_wide
        )

        init_fn, apply_fn, kernel_fn = stax.serial(
            two_wl_aggregation_layer,
            two_wl_aggregation_layer,
            two_wl_aggregation_layer,
            stax.GlobalSumPool(),
            stax.Dense(1),
        )

        return init_fn, apply_fn, kernel_fn

    else:
        raise Exception(
            f"get_network_configuration has no case for dataset: {dataset} method: {method}"
        )
