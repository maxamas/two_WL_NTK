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


def check_if_output_allready_exists(type: str, dataset_path: str) -> bool:
    def check(files: List[str]) -> bool:
        all_not_exist = True
        for f in files:
            all_not_exist = all_not_exist and not os.path.exists(dataset_path + f)

        return not all_not_exist

    files_TWL = [
        "/two_wl_edge_features.jnpy",
        "/two_wl_patterns.jnpy",
        "/two_wl_ys.jnpy",
        "/two_wl_nb_edges.jnpy",
        "/two_wl_patterns_graph_map.jnpy",
    ]

    files_GCN = [
        "/gcn_sparse_node_features",
        "/gcn_sparse_patterns",
        "/gcn_sparse_ys",
        "/gcn_sparse_nb_nodes",
        "/gcn_sparse_patterns_graph_map",
    ]

    if type == "TWL":
        return check(files_TWL)
    elif type == "GCN":
        return check(files_GCN)
    else:
        raise Exception(
            f"Type {type} not valid for check_if_output_allready_exists! Type must be one of TWL, GCN!"
        )


def prepare_tu_data_for_2WL(
    tu_datasets: List[str], base_path: str, base_path_tu_datasets: str
):

    for dataset_name in tu_datasets:
        dataset_path = base_path + f"/{dataset_name}"

        print(f"Preparing {dataset_name} dataset:")
        if check_if_output_allready_exists("TWL", dataset_path):
            print(
                f"All expected output files already exist at {dataset_path}. If you want to preproces again delete the files at that location. Skip the Dataset for now."
            )
            continue

        try:
            dataset = TUDataset(root=base_path_tu_datasets, name=dataset_name)
        except Exception as e:
            print(f"Can not download datset. Error {e}")

        dataset_edge_features = list()
        dataset_patterns = list()
        dataset_ys = list()
        dataset_nb_edges = list()
        patterns_graph_map = list()
        prev_pattern_graph_map = 0
        for i, data in enumerate(dataset):
            if i % 10 == 0:
                print(f"Working on sample {i} to {i + 10} of {len(dataset)} samples.")

            # calculate the edge_list and pattern array for each graph in the dataset
            edge_list = jnp.transpose(jnp.array(data.edge_index))
            node_features = jnp.array(data.x)
            nb_nodes = len(node_features)
            pattern, edge_list = twl_sparse_pattern(edge_list, nb_nodes)
            edge_features = twl_sparse_edge_features(
                node_features, edge_list.shape[0], nb_nodes
            )

            dataset_edge_features.append(edge_features)
            dataset_patterns.append(pattern)
            dataset_ys.append(jnp.array(data.y))
            dataset_nb_edges.append(edge_list.shape[0])
            patterns_graph_map.append(pattern.shape[0] + prev_pattern_graph_map)
            prev_pattern_graph_map = patterns_graph_map[-1]

        # merge all patterns and all edge lists into one big pattern
        # and move the indices in pattern acordingly
        pattern = dataset_patterns[0]
        nb_edges_cum = dataset_nb_edges[0]
        for current_nb_edges, current_pattern in zip(
            dataset_nb_edges[1:], dataset_patterns[1:]
        ):
            pattern = jnp.append(pattern, current_pattern + nb_edges_cum, 0)
            nb_edges_cum += current_nb_edges

        edge_features = dataset_edge_features[0]
        for current_edge_features in dataset_edge_features[1:]:
            edge_features = jnp.append(edge_features, current_edge_features, 0)

        if not os.path.exists(dataset_path):
            print(f"Creating directory: {dataset_path}")
            os.makedirs(dataset_path)

        print(f"Saving output files at: {dataset_path}")

        jnp.save(dataset_path + f"/two_wl_edge_features", edge_features)
        jnp.save(dataset_path + f"/two_wl_patterns", pattern)
        jnp.save(dataset_path + f"/two_wl_ys", jnp.array(dataset_ys))
        jnp.save(dataset_path + f"/two_wl_nb_edges", jnp.array(dataset_nb_edges))
        jnp.save(
            dataset_path + f"/two_wl_patterns_graph_map", jnp.array(patterns_graph_map)
        )


def prepare_tu_data_for_GCN(
    tu_datasets: List[str], base_path: str, base_path_tu_datasets: str
):

    for dataset_name in tu_datasets:
        dataset_path = base_path + f"/{dataset_name}"

        print(f"Preparing {dataset_name} dataset:")
        if check_if_output_allready_exists("GCN", dataset_path):
            print(
                f"All expected output files already exist at {dataset_path}. If you want to preproces again delete the files at that location. Skip the Dataset for now."
            )
            continue
        try:
            dataset = TUDataset(root=base_path_tu_datasets, name=dataset_name)
        except Exception as e:
            print(f"Can not download datset. Error {e}")
            continue

        dataset_node_features = list()
        dataset_patterns = list()
        dataset_ys = list()
        dataset_nb_nodes = list()
        patterns_graph_map = list()
        prev_pattern_graph_map = 0
        for i, data in enumerate(dataset):
            if i % 10 == 0:
                print(f"Working on sample {i} to {i + 10} of {len(dataset)} samples.")

            edge_list = jnp.transpose(jnp.array(data.edge_index))
            node_features = jnp.array(data.x)
            nb_nodes = len(node_features)

            dataset_node_features.append(node_features)
            dataset_patterns.append(edge_list)
            dataset_ys.append(jnp.array(data.y))
            dataset_nb_nodes.append(nb_nodes)
            patterns_graph_map.append(node_features.shape[0] + prev_pattern_graph_map)
            prev_pattern_graph_map = patterns_graph_map[-1]

        # merge all patterns and all edge lists into one big pattern
        # and move the indices in pattern acordingly
        pattern = dataset_patterns[0]
        nb_nodes_cum = dataset_nb_nodes[0]
        for current_nb_nodes, current_pattern in zip(
            dataset_nb_nodes[1:], dataset_patterns[1:]
        ):
            pattern = jnp.append(pattern, current_pattern + nb_nodes_cum, 0)
            nb_nodes_cum += current_nb_nodes

        node_features = dataset_node_features[0]
        for current_node_features in dataset_node_features[1:]:
            node_features = jnp.append(node_features, current_node_features, 0)

        # node feature
        node_features = jnp.expand_dims(node_features, 0)
        node_features = jnp.expand_dims(node_features, 2)

        # pattern
        pattern = jnp.expand_dims(pattern, 0)
        pattern = jnp.expand_dims(pattern, 2)

        if not os.path.exists(dataset_path):
            print(f"Creating directory: {dataset_path}")
            os.makedirs(dataset_path)

        print(f"Saving output files at: {dataset_path}")
        jnp.save(dataset_path + f"/gcn_sparse_node_features", node_features)
        jnp.save(dataset_path + f"/gcn_sparse_patterns", pattern)
        jnp.save(dataset_path + f"/gcn_sparse_ys", jnp.array(dataset_ys))
        jnp.save(dataset_path + f"/gcn_sparse_nb_nodes", jnp.array(dataset_nb_nodes))
        jnp.save(
            dataset_path + f"/gcn_sparse_patterns_graph_map",
            jnp.array(patterns_graph_map),
        )


if __name__ == "__main__":
    # tu_datasets = [
    #     "MUTAG",
    #     "PROTEINS",
    #     "PTC",
    #     "NCI1",
    #     "COLLAB",
    #     "IMDB-BINARY",
    #     "IMDB-MULTI",
    # ]

    tu_datasets = ["MUTAG"]
    base_path_preprocessed = f"~/masterarbeit/MasterarbeitData/Preprocessed"
    base_path_tu_datasets = f"~/masterarbeit/MasterarbeitData/TUData"

    # prepare tudatasets for 2WL
    prepare_tu_data_for_2WL(tu_datasets, base_path_preprocessed, base_path_tu_datasets)

    # prepare tu datasets for sparse gcn
    prepare_tu_data_for_GCN(tu_datasets, base_path_preprocessed, base_path_tu_datasets)
