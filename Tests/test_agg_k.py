# copy the kernel function from the layers module

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
from utils import row_wise_karthesian_prod
from jax import numpy as jnp
from jax import numpy as np
import jax
import neural_tangents as nt


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


toy_kernel_2_new = np.array(
    [
        [1, 2, 3, 4, 5],
        [2, 1, 3, 4, 5],
        [3, 3, 1, 4, 5],
        [4, 4, 4, 1, 5],
        [5, 5, 5, 5, 1],
    ]
)

toy_kernel_2_new_indx = jnp.array([0, 0, 0, 1, 1])

print(toy_kernel_2_new)
print(toy_kernel_2_new.shape)


k = nt.Kernel(
    nngp=toy_kernel_2_new,
    ntk=toy_kernel_2_new,
    cov1=None,
    cov2=None,
    x1_is_x2=None,
    is_gaussian=None,
    is_reversed=None,
    is_input=None,
    diagonal_batch=None,
    diagonal_spatial=None,
    shape1=None,
    shape2=None,
    batch_axis=None,
    channel_axis=None,
    mask1=None,
    mask2=None,
)

kernel_matrix = kernel_fn(
    k=k, graph_indx=(toy_kernel_2_new_indx, toy_kernel_2_new_indx), nb_graphs=(2, 2)
)

print(jnp.sum(toy_kernel_2_new[:3, :3]))
print(jnp.sum(toy_kernel_2_new[3:, 3:]))
print(jnp.sum(toy_kernel_2_new[:3, 3:]))
print(jnp.sum(toy_kernel_2_new[3:, :3]))


print(kernel_matrix.ntk)
