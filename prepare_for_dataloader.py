from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from jax import numpy as jnp
import os
import config
import os
from utils import row_wise_karthesian_prod
from torch_geometric.datasets import TUDataset
from jax import numpy as jnp
from jax._src.typing import Array
from typing import Tuple, List


def ravel_multi_index_upper(a: Array, b: Array, a_dim: int, b_dim: int) -> Array:
    """
    for 2 dim array.
    Like jnp.ravel_multi_index but for the elements in the lower triangle
    return the index of the element ont the transposed array.
    """
    a_tmp = jnp.copy(a)
    a_tmp = a_tmp.at[a > b].set(b[a > b])
    b = b.at[a > b].set(a[a > b])

    return jnp.ravel_multi_index([a_tmp, b], (a_dim, b_dim))


def twl_sparse_pattern(edge_list: Array, nb_nodes: int) -> Tuple[Array, Array]:
    # remove all edges which would be on the lower triangle and on the diag
    edges_list = edge_list[edge_list[:, 0] < edge_list[:, 1]]

    # sort
    ind = jnp.lexsort((edges_list[:, 1], edges_list[:, 0]), 0)
    edge_list = edges_list[ind, :]

    # add self edges (we removed the first, to make the to sort array
    # smaller and to make sure the self edges are on the top of the edge_list)
    self_edges = jnp.array([[i, i] for i in range(0, nb_nodes)])
    edge_list = jnp.append(self_edges, edge_list, 0)

    # consider the not self edges also in reversed order
    edge_list_full = jnp.append(edge_list, edge_list[nb_nodes:, [1, 0]], 0)

    # karthesian product
    edge_list_full = row_wise_karthesian_prod(edge_list_full, edge_list_full)

    # only edges where i and j have same neighbours
    edge_list_full = edge_list_full[edge_list_full[:, 1] == edge_list_full[:, 2]]

    # add the i, j nodes and select only elemnt on the upper triangle based on i,j
    edge_list_full = jnp.append(edge_list_full[:, [0, -1]], edge_list_full, 1)
    edge_list_full = edge_list_full[edge_list_full[:, 0] <= edge_list_full[:, 1]]

    # need this to index the edge feature matrix
    linear_indx = ravel_multi_index_upper(
        edge_list[:, 0], edge_list[:, 1], nb_nodes, nb_nodes
    )
    edges_map_full = jnp.full(
        (edge_list_full.shape[0], linear_indx.shape[0]), linear_indx
    )

    # calculate the indices on the theoratical dense matrix
    # where indices on the (theoretical) lower triangle are
    # swapped with upper triangle indices
    a = ravel_multi_index_upper(
        edge_list_full[:, 0], edge_list_full[:, 1], nb_nodes, nb_nodes
    )
    b = ravel_multi_index_upper(
        edge_list_full[:, 2], edge_list_full[:, 3], nb_nodes, nb_nodes
    )
    c = ravel_multi_index_upper(
        edge_list_full[:, 4], edge_list_full[:, 5], nb_nodes, nb_nodes
    )

    # map the dense indices to the actual indices of the spars edge list
    a = jnp.argmax(edges_map_full == jnp.expand_dims(a, 1), 1)
    b = jnp.argmax(edges_map_full == jnp.expand_dims(b, 1), 1)
    c = jnp.argmax(edges_map_full == jnp.expand_dims(c, 1), 1)

    pattern = jnp.array([a, b, c]).transpose()

    # sorting only for debuging
    # ind = jnp.lexsort((pattern[:, 1], pattern[:, 0]), 0)
    # pattern = pattern[ind, :]

    return pattern, edge_list


def twl_sparse_edge_features(
    node_features: Array, nb_edges: int, nb_nodes: int
) -> Array:

    edge_features = jnp.zeros((nb_edges, node_features.shape[1] + 1))
    edge_features = edge_features.at[:nb_nodes, :-1].set(node_features)
    edge_features = edge_features.at[:, -1].set(1)

    return edge_features


def prepare_gcn_dataset(
    dataset_name: str, tu_datasets_path: str, preprocessed_path: str
):

    dataset = TUDataset(root=tu_datasets_path, name=dataset_name)

    for i, data in enumerate(dataset):
        edge_list = jnp.transpose(jnp.array(data.edge_index))
        node_features = jnp.array(data.x)
        sample_path = preprocessed_path + f"/gcn_id_{i}"
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        jnp.save(sample_path + "/node_features.npy", node_features)
        jnp.save(sample_path + "/edge_list.npy", edge_list)
        jnp.save(sample_path + "/y.npy", jnp.array(data.y))


def prepare_twl_dataset(
    dataset_name: str, tu_datasets_path: str, preprocessed_path: str
):

    dataset = TUDataset(root=tu_datasets_path, name=dataset_name)

    for i, data in enumerate(dataset):

        edge_list = jnp.transpose(jnp.array(data.edge_index))
        node_features = jnp.array(data.x)
        sample_path = preprocessed_path + f"/twl_id_{i}"
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        nb_nodes = len(data.x)
        ref_matrix, edge_list = twl_sparse_pattern(edge_list, nb_nodes)
        edge_features = twl_sparse_edge_features(
            node_features, edge_list.shape[0], nb_nodes
        )

        jnp.save(sample_path + "/edge_features.npy", edge_features)
        jnp.save(sample_path + "/ref_matrix.npy", ref_matrix)
        jnp.save(sample_path + "/y.npy", jnp.array(data.y))


if __name__ == "__main__":

    tu_datasets = config.dataset_names

    for dataset_name in tu_datasets:
        print(f"Preparing Dataset {dataset_name} for GCN: ")
        prepare_gcn_dataset(
            dataset_name,
            config.base_path_tu_datasets,
            config.dataloader_base_path + f"/{dataset_name}/GCN",
        )
        print(f"Preparing Dataset {dataset_name} for TWL: ")
        prepare_twl_dataset(
            dataset_name,
            config.base_path_tu_datasets,
            config.dataloader_base_path + f"/{dataset_name}/TWL",
        )
