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
from typing import Tuple, List, Callable
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, List


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


def resort_edge_features(edge_list: Array, edge_features: Array, nb_nodes: int):
    """
    Remove the edge features with i < j. The edge son the lower diag of 
    the adjacency matrix. (Assume symetric Adjaceny matrix)
    Sort the self edges to the top of the edge_features array.
    """
    el_ef = jnp.append(edge_list, edge_features, 1)

    self_edges = el_ef[el_ef[:, 0] == el_ef[:, 1]]
    ind = jnp.lexsort((self_edges[:, 1], self_edges[:, 0]), 0)
    self_edges = self_edges[ind, :]

    if self_edges.shape[0] < nb_nodes:
        self_edges = jnp.zeros((nb_nodes, el_ef.shape[1]-2))
        self_edges = self_edges.at[:,0].set(1)
        edge_list_self = jnp.full((nb_nodes, 2), jnp.expand_dims(jnp.array(range(nb_nodes)), 1))
        self_edges = jnp.append(edge_list_self, self_edges, 1)

    el_ef = el_ef[el_ef[:, 0] < el_ef[:, 1]]
    ind = jnp.lexsort((el_ef[:, 1], el_ef[:, 0]), 0)
    el_ef = el_ef[ind, :]

    out = jnp.array(jnp.append(self_edges, el_ef, 0), "int32")
    
    return out[:,:2], out[:,2:]


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


def append_node_edge_features(node_features: Array, edge_features: Array):
    out = jnp.zeros(
        (edge_features.shape[0], node_features.shape[1] + edge_features.shape[1])
    )
    out = out.at[: node_features.shape[0], : node_features.shape[1]].set(node_features)
    out = out.at[:, node_features.shape[1] :].set(edge_features)
    return out


def prepare_gcn_dataset(
    dataset_name: str, tu_datasets_path: str, preprocessed_path: str
):

    try:
        dataset = TUDataset(root=tu_datasets_path, name=dataset_name)
    except Exception as e:
        print(f"Can not download datset. Error {e}")
        return None

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

    try:
        dataset = TUDataset(root=tu_datasets_path, name=dataset_name)
    except Exception as e:
        print(f"Can not download datset. Error {e}")
        return None

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


def save_gcn_sample(
    sample_id: int,
    preprocessed_path: str,
    y: Array,
    edge_list: Array,
    node_features: Array,
):
    sample_path = preprocessed_path + f"/gcn_id_{sample_id}"
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    jnp.save(sample_path + "/node_features.npy", node_features)
    jnp.save(sample_path + "/edge_list.npy", edge_list)
    jnp.save(sample_path + "/y.npy", y)


def save_twl_sample(
    sample_id: int,
    preprocessed_path: str,
    y: Array,
    edge_list: Array,
    node_features: Array,
    edge_features: Array,
):

    nb_nodes = node_features.shape[0]
    edge_list_n, edge_features_n  = resort_edge_features(edge_list, edge_features, nb_nodes)
    ref_matrix, edge_list = twl_sparse_pattern(edge_list, nb_nodes)
    # TODO: Need to reorder the edge_features! The self edges need to be on the top of the edge_features array
    edge_features = append_node_edge_features(node_features, edge_features_n)

    if jnp.all(jnp.logical_not(edge_list_n == edge_list)):
        print("Edge list sorting is different")
        exit()

    sample_path = preprocessed_path + f"/twl_id_{sample_id}"
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    jnp.save(sample_path + "/edge_features.npy", edge_features)
    jnp.save(sample_path + "/ref_matrix.npy", ref_matrix)
    jnp.save(sample_path + "/y.npy", y)


def prepare_datasets(
    dataset_name: str,
    tu_datasets_path: str,
    preprocessed_path: str,
    make_node_features: Callable,
    make_edge_features: Callable,
):

    dataset = TUDataset(root=tu_datasets_path, name=dataset_name)

    has_node_features: bool = dataset.num_node_features > 0
    has_edge_features: bool = dataset.num_edge_features > 0

    # has_node_features: bool = not dataset[0].x == None
    # has_edge_features: bool = not dataset[0].edge_attr == None

    nb_nodes = [data.edge_index.max() + 1 for data in dataset]
    nb_edges = [data.edge_index.shape[1] for data in dataset]

    max_nb_nodes = max(nb_nodes)
    max_nb_edges = max(nb_edges)

    for sample_id, data in enumerate(dataset):

        edge_list = jnp.transpose(jnp.array(data.edge_index))
        y = jnp.array(data.y)

        if has_node_features:
            node_features = jnp.array(data.x)
        else:
            node_features = make_node_features(
                max_nb_nodes, list(range(nb_nodes[sample_id]))
            )

        if has_edge_features:
            edge_features = jnp.array(data.edge_attr)
        else:
            edge_features = make_edge_features(
                max_nb_edges, list(range(nb_edges[sample_id]))
            )

        save_twl_sample(
            sample_id,
            preprocessed_path + "/TWL",
            y,
            edge_list,
            node_features,
            edge_features,
        )

        save_gcn_sample(
            sample_id,
            preprocessed_path + "/GCN",
            y,
            edge_list,
            node_features,
        )


def bias_edges(max_edges: int, edges_indxs: List[int]):
    return jnp.ones((len(edges_indxs),1))


def one_hot_nodes(max_nodes: int, nodes_indxs: List[int]):
    return jnp.eye(max_nodes, max_nodes)[nodes_indxs, :]


def one_hot_edges(max_edges: int, edges_indxs: List[int]):
    return jnp.eye(max_edges, max_edges)[edges_indxs, :]


if __name__ == "__main__":

    # tu_datasets = config.dataset_names

    # for dataset_name in tu_datasets:
    #     print(f"Preparing Dataset {dataset_name} for GCN: ")
    #     dataload_base_path = config.dataloader_base_path + f"/{dataset_name}/GCN"
    #     if not os.path.exists(dataload_base_path):
    #         try:
    #             prepare_gcn_dataset(
    #                 dataset_name,
    #                 config.base_path_tu_datasets,
    #                 dataload_base_path,
    #             )
    #         except Exception as e:
    #             print(f"Preparing Dataset {dataset_name} for GCN failed! Error {e}")
    #     else:
    #         print(
    #             f"Folder {dataload_base_path} already exits. Skip Datset {dataset_name} for GCN for now. Delete the folder, if you want to rerun the data preperation!"
    #         )
    #     print(f"Preparing Dataset {dataset_name} for TWL: ")
    #     dataload_base_path = config.dataloader_base_path + f"/{dataset_name}/TWL"
    #     if not os.path.exists(dataload_base_path):
    #         try:
    #             prepare_twl_dataset(
    #                 dataset_name,
    #                 config.base_path_tu_datasets,
    #                 dataload_base_path,
    #             )
    #         except Exception as e:
    #             print(f"Preparing Dataset {dataset_name} for TWL failed! Error {e}")
    #     else:
    #         print(
    #             f"Folder {dataload_base_path} already exits. Skip Datset {dataset_name} for TWL for now. Delete the folder, if you want to rerun the data preperation!"
    #         )

    # import torch_geometric.transforms as T
    # from torch_geometric.datasets import TUDataset

    # dataset_name = "PTC_MR"
    # dataload_base_path = config.dataloader_base_path + f"/{dataset_name}/GCN"
    # dataset = TUDataset(root=config.base_path_tu_datasets, name=dataset_name, use_node_attr=True, use_edge_attr=True)

    # dataset_name = "COLORS-3"
    # dataset2 = TUDataset(root=config.base_path_tu_datasets, name=dataset_name, use_node_attr=True, use_edge_attr=True)

    # tu_datasets = config.dataset_names

    # for dataset_name in tu_datasets:
    #     print("Dataset: ", dataset_name)
    #     d = TUDataset(root=config.base_path_tu_datasets, name=dataset_name, use_node_attr=True, use_edge_attr=True)
    #     print("Node features is None: ", d[0].x == None)
    #     print("Edge features is None: ", d[0].edge_attr == None)
    #     print("Pos is None: ", d[0].pos == None)

    # dataset_name = "MUTAG"
    # dataset = TUDataset(
    #     root=config.base_path_tu_datasets,
    #     name=dataset_name,
    #     use_node_attr=True,
    #     use_edge_attr=True,
    # )

    # tu_datasets = ["MUTAG"]
    tu_datasets = ["MUTAG", "COLORS-3"]


    for dataset_name in tu_datasets:
        print(f"Preparing Dataset {dataset_name} for GCN and TWL: ")
        dataload_base_path = config.dataloader_base_path + f"/{dataset_name}"
        if not os.path.exists(dataload_base_path):
            # try:
            prepare_datasets(
                dataset_name,
                config.base_path_tu_datasets,
                dataload_base_path,
                one_hot_nodes,
                bias_edges,
            )

            # except Exception as e:
            #     print(f"Preparing Dataset {dataset_name} for GCN failed! Error {e}")
        else:
            print(
                f"Folder {dataload_base_path} already exits. Skip Datset {dataset_name} for GCN for now. Delete the folder, if you want to rerun the data preperation!"
            )
