from layers import get_two_wl_aggregation_layer, index_aggregation, gcn_aggregation
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


def get_2wl_network_configuration(
    layers, parameterization, layer_wide, output_layer_wide
):
    two_wl_aggregation_layer = get_two_wl_aggregation_layer(
        parameterization, layer_wide
    )

    layers = tuple(tuple(two_wl_aggregation_layer) for i in range(layers)) + (
        tuple(index_aggregation()),
        tuple(stax.Dense(output_layer_wide)),
    )

    return stax.serial(*layers)  # init_fn, apply_fn, kernel_fn


def get_gcn_network_configuration(
    layers, parameterization, layer_wide, output_layer_wide
):
    gcn_layer = stax.serial(
        stax.Dense(layer_wide, parameterization=parameterization),
        stax.Relu(),
        gcn_aggregation(),
    )

    layers = tuple(tuple(gcn_layer) for i in range(layers)) + (
        tuple(index_aggregation()),
        tuple(stax.Dense(output_layer_wide)),
    )

    return stax.serial(*layers)  # init_fn, apply_fn, kernel_fn
