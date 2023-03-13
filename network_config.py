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


def get_2wl_network_configuration(
    layers, parameterization, layer_wide, output_layer_wide
):
    two_wl_aggregation_layer = get_two_wl_aggregation_layer(
        parameterization, layer_wide
    )

    layers = tuple(tuple(two_wl_aggregation_layer) for i in range(layers)) + (
        tuple(stax.GlobalSumPool()),
        tuple(stax.Dense(output_layer_wide)),
    )

    return stax.serial(*layers)  # init_fn, apply_fn, kernel_fn


def get_gcn_network_configuration(
    layers, parameterization, layer_wide, output_layer_wide
):
    gcn_layer = stax.serial(
        stax.Conv(layer_wide, (1, 1), parameterization=parameterization),
        stax.Relu(),
        stax.Aggregate(
            aggregate_axis=1, batch_axis=0, channel_axis=3, implementation="SPARSE"
        ),
    )

    layers = tuple(tuple(gcn_layer) for i in range(layers)) + (
        tuple(stax.GlobalSumPool()),
        tuple(stax.Dense(output_layer_wide)),
    )

    return stax.serial(*layers)  # init_fn, apply_fn, kernel_fn
