import os
from jax import numpy as jnp
from dataloader import GCN_Dataloader, TWL_Dataloader, Dataloader
import config
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, List
import jax
from jax._src.typing import Array
from network_config import get_2wl_network_configuration, get_gcn_network_configuration


def twl_kernel_function(kernel_fn, type="ntk"):
    # x and y are tuples with:
    # edge_features, ref_matrix, edge_features_graph_indx, ref_matrix_graph_indx, ys, nb_edges, nb_graphs, id_high
    return lambda x, y: kernel_fn(
        x[0],
        y[0],
        type,
        pattern=(x[1], y[1]),
        nb_edges=(x[0].shape[0], y[0].shape[0]),
        nb_graphs=(x[6], y[6]),
        graph_indx=(x[2], y[2]),
    )


def gcn_kernel_function(kernel_fn, type="ntk"):
    # x and y are tuples with:
    # (node_features, edge_list, node_features_graph_indx, edge_list_graph_indx, ys, nb_nodes, nb_graphs, id_high)
    return lambda x, y: kernel_fn(
        x[0],
        y[0],
        type,
        pattern=(x[1], y[1]),
        nb_nodes=(x[0].shape[0], y[0].shape[0]),
        nb_graphs=(x[6], y[6]),
        graph_indx=(x[2], y[2]),
    )


def save_gram_matrix_batch_wise(
    batch_iterator_1: Callable, batch_iterator_2: Callable, kernel_fn, kernel_path: str
):
    for i, arrays_1 in enumerate(batch_iterator_1()):
        for j, arrays_2 in enumerate(batch_iterator_2()):
            kernel_matrix = kernel_fn(arrays_1, arrays_2)
            if not os.path.exists(kernel_path):
                os.makedirs(kernel_path)
            jnp.save(kernel_path + f"/NTK_{arrays_1[-1]}_{arrays_2[-1]}", kernel_matrix)
        jnp.save(kernel_path + f"/Ys_2_{arrays_1[-1]}", arrays_1[4])


def save_kernels(datasets_names: List[str], nn_types: List[str]):
    for dataset_name in datasets_names:
        for nn_type in nn_types:

            base_path_preprocessed = (
                config.dataloader_base_path + f"/{dataset_name}/{nn_type}"
            )
            kernel_path = config.kernel_base_path + f"/{dataset_name}/{nn_type}"

            if nn_type == "GCN":
                data_loader = GCN_Dataloader(
                    file_path=base_path_preprocessed, nb_train_samples=160
                )
                _, _, kernel_fn = get_gcn_network_configuration(
                    layers=10,
                    parameterization="ntk",
                    layer_wide=10,
                    output_layer_wide=1,
                )
                decorated_kernel_fn = gcn_kernel_function(kernel_fn, type="ntk")
            elif nn_type == "TWL":
                data_loader = TWL_Dataloader(
                    file_path=base_path_preprocessed, nb_train_samples=160
                )
                # layer with must be specified as int, but is ignored
                _, _, kernel_fn = get_2wl_network_configuration(
                    layers=3, parameterization="ntk", layer_wide=10, output_layer_wide=1
                )
                decorated_kernel_fn = twl_kernel_function(kernel_fn, type="ntk")
            else:
                print(f"No path for nn_type {nn_type}")
                exit()

            # create a batch iterator for all samples yielded in an ordered way
            data_it_1 = lambda: data_loader.batch_iterator(5, True)
            data_it_2 = lambda: data_loader.batch_iterator(5, True)

            save_gram_matrix_batch_wise(
                data_it_1, data_it_2, decorated_kernel_fn, kernel_path
            )


if __name__ == "__main__":

    datasets_names = config.dataset_names
    nn_types = config.nn_types

    save_kernels(datasets_names, nn_types)
