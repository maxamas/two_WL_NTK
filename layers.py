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
from utils import row_wise_karthesian_prod
from neural_tangents import stax

# pattern is a two dimensional array of shape k x 4
# a edge list from the 4 dimensional intersected adjacency matrix
# R = np.transpose(np.array(np.nonzero(As_int)))
# As_int is four dimensional with batch x #nodes x #nodes x #nodes
# e.g. As_int[a,b,c,:] gives all nodes, which are neighboors
# of nodes b and c in batch a.

# Thus e.g a row from the pattern array [A,B,C,D] tells us,
# that node D is a neighboor of nodes B and C in batch A.


# TODO: remove the n_nodes argument


@layer
@supports_masking(remask_kernel=False)
def gcn_aggregation():

    init_fn = lambda rng, input_shape: (input_shape, ())

    def apply_fn(
        params, inputs: np.ndarray, *, pattern: Optional[np.ndarray] = None, **kwargs
    ):

        sum_nodes = jax.ops.segment_sum(
            np.take(inputs, pattern[:, 1], axis=0), pattern[:, 0], inputs.shape[0]
        )

        scalar = jax.ops.segment_sum(
            np.ones(pattern[:, 0].shape[0]), pattern[:, 0], inputs.shape[0]
        )
        out = np.divide(sum_nodes, np.expand_dims(scalar, 1))
        return out

    def kernel_fn(
        k: Kernel,
        *,
        pattern: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
        nb_nodes: Tuple[Optional[int], Optional[int]] = (None, None),
        **kwargs
    ):

        num_segments = int(np.prod(np.array(k.ntk.shape)))

        patterns = row_wise_karthesian_prod(pattern[0], pattern[1])

        nodes_v_vb = np.ravel_multi_index(
            [patterns[:, 0], patterns[:, 2]], (nb_nodes[0], nb_nodes[1])
        )
        nodes_u_ub = np.ravel_multi_index(
            [patterns[:, 1], patterns[:, 3]], (nb_nodes[0], nb_nodes[1])
        )

        def agg(x):
            kernel = jax.ops.segment_sum(
                np.take(x, nodes_u_ub), nodes_v_vb, num_segments
            )
            scaler = jax.ops.segment_sum(
                np.ones(nodes_v_vb.shape[0]), nodes_v_vb, num_segments
            )
            scaler = scaler + np.full(scaler.shape, 2)
            kernel = np.divide(kernel, scaler)
            kernel = np.reshape(kernel, x.shape)
            return kernel

        ntk = agg(k.ntk)
        nngp = agg(k.nngp)

        return k.replace(ntk=ntk, nngp=nngp, is_gaussian=True, is_input=False)

    return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def two_wl_aggregation():
    """
    Return a layer, that implements the gatter
    and scatter operations given the reference Matrix
    R.
    R is of shape (#graps in the batch *
    # edges of each grap) x 2
    """

    init_fn = lambda rng, input_shape: (input_shape, ())

    def apply_fn(
        params, inputs: np.ndarray, *, pattern: Optional[np.ndarray] = None, **kwargs
    ):

        sum_edges_i_a = jax.ops.segment_sum(
            np.take(inputs, pattern[:, 1], axis=0), pattern[:, 0], inputs.shape[0]
        )
        sum_edges_a_j = jax.ops.segment_sum(
            np.take(inputs, pattern[:, 2], axis=0), pattern[:, 0], inputs.shape[0]
        )

        return sum_edges_i_a + sum_edges_a_j

    def kernel_fn(
        k: Kernel,
        *,
        pattern: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
        nb_edges: Tuple[Optional[int], Optional[int]] = (None, None),
        **kwargs
    ):

        num_segments = int(np.prod(np.array(k.ntk.shape)))

        patterns = row_wise_karthesian_prod(pattern[0], pattern[1])

        e_i_j_ib_jb = np.ravel_multi_index(
            [patterns[:, 0], patterns[:, 3]], (nb_edges[0], nb_edges[1])
        )
        e_i_a_ib_ab = np.ravel_multi_index(
            [patterns[:, 1], patterns[:, 4]], (nb_edges[0], nb_edges[1])
        )
        e_i_a_ab_jb = np.ravel_multi_index(
            [patterns[:, 1], patterns[:, 5]], (nb_edges[0], nb_edges[1])
        )
        e_a_j_ib_ab = np.ravel_multi_index(
            [patterns[:, 2], patterns[:, 4]], (nb_edges[0], nb_edges[1])
        )
        e_a_j_ab_jb = np.ravel_multi_index(
            [patterns[:, 2], patterns[:, 5]], (nb_edges[0], nb_edges[1])
        )

        def agg(x):
            theta_i_a_ib_ab = jax.ops.segment_sum(
                np.take(x, e_i_a_ib_ab), e_i_j_ib_jb, num_segments
            )
            theta_i_a_ab_jb = jax.ops.segment_sum(
                np.take(x, e_i_a_ab_jb), e_i_j_ib_jb, num_segments
            )
            theta_a_j_ib_ab = jax.ops.segment_sum(
                np.take(x, e_a_j_ib_ab), e_i_j_ib_jb, num_segments
            )
            theta_a_j_ab_jb = jax.ops.segment_sum(
                np.take(x, e_a_j_ab_jb), e_i_j_ib_jb, num_segments
            )

            thetas_linear = np.array(
                [theta_i_a_ib_ab, theta_i_a_ab_jb, theta_a_j_ib_ab, theta_a_j_ab_jb]
            )
            theta_linear = np.sum(thetas_linear, 0)
            theta = np.reshape(theta_linear, x.shape)
            return theta

        ntk = agg(k.ntk)
        nngp = agg(k.nngp)

        return k.replace(ntk=ntk, nngp=nngp, is_gaussian=True, is_input=False)

    return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def index_aggregation():
    init_fn = lambda rng, input_shape: (input_shape, ())

    def apply_fn(
        params, inputs: np.ndarray, *, graph_indx: Optional[np.ndarray] = None, **kwargs
    ):

        # Can not use nb_graphs here as this gives an error when
        # jiting the apply function for gradient descent
        # thus need to use the input shape.
        # (the num_segments argument must be known at complie time
        # and can not be dynamic)
        # This implies, that the output shape is of the same
        # size as the input.
        # Only the first nb_graphs entries of the output are non 0 thoug.
        # The hack to make this work is to only consider the
        # first nb_graphs entires for the loss calculation
        num_segments = inputs.shape[0]
        out = np.apply_along_axis(
            lambda x: jax.ops.segment_sum(x, graph_indx, num_segments),
            0,
            np.squeeze(inputs),
        )
        return out

    def kernel_fn(
        k: Kernel,
        *,
        graph_indx: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None),
        nb_graphs: Tuple[Optional[int], Optional[int]] = (None, None),
        **kwargs
    ):
        num_segments = nb_graphs[0] * nb_graphs[1]

        def agg(x, kernel_graph_indx):
            agg_x = jax.ops.segment_sum(
                np.reshape(x, (-1)),
                kernel_graph_indx,
                num_segments,
            )
            agg_x = np.reshape(agg_x, nb_graphs)
            agg_x = np.expand_dims(agg_x, 0)
            agg_x = np.expand_dims(agg_x, 0)
            return agg_x

        k_prod_graph_indx = row_wise_karthesian_prod(
            np.expand_dims(graph_indx[0], 1), np.expand_dims(graph_indx[1], 1)
        )
        kernel_graph_indx = np.ravel_multi_index(
            [k_prod_graph_indx[:, 0], k_prod_graph_indx[:, 1]], nb_graphs
        )

        agg_ntk = agg(k.ntk, kernel_graph_indx)
        agg_nngp = agg(k.nngp, kernel_graph_indx)

        return k.replace(
            ntk=agg_ntk, nngp=agg_nngp, is_gaussian=True, is_input=False, channel_axis=1
        )

    return init_fn, apply_fn, kernel_fn


def get_two_wl_aggregation_layer(parameterization, layer_wide):
    """
    parameterization: "standard" or "ntk"
    n_nodes: the max number of nodes in all graphs
    layer_wide: only relevant if the ntk is derived
    """
    L_branche = stax.serial(
        stax.Dense(layer_wide, parameterization=parameterization),
    )

    Gamma_branche = stax.serial(
        stax.Dense(layer_wide, parameterization=parameterization),
        two_wl_aggregation(),
    )

    two_wl_aggregation_layer = stax.serial(
        stax.FanOut(2),
        stax.parallel(L_branche, Gamma_branche),
        stax.FanInSum(),
        stax.Relu(),
    )

    return two_wl_aggregation_layer
