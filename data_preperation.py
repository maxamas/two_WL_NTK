from utils import *

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import neural_tangents as nt
from neural_tangents import stax
import jax
from jax import numpy as jnp

from jax._src.typing import Array, Shape
from typing import Callable, Tuple, List


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
    ind = jnp.lexsort((pattern[:, 1], pattern[:, 0]), 0)
    pattern = pattern[ind, :]

    return pattern, edge_list


def twl_sparse_edge_features(
    node_features: Array, nb_edges: int, nb_nodes: int
) -> Array:

    edge_features = jnp.zeros((nb_edges, node_features.shape[1] + 1))
    edge_features = edge_features.at[:nb_nodes, :-1].set(node_features)
    edge_features = edge_features.at[:, -1].set(1)

    return edge_features


def initial_edge_features(
    graps_node_features, graphs_edge_features, edge_featur_init, nb_graphs, max_nodes
):
    """
    get the initial edge feature matrix for the 2 WL
    algorithm.
    when the graph specification has only node features
    as a batch x nodes x nodes x channels array
    """
    feature_dim = graps_node_features[0][0].shape[0]
    # feature_dim = graps_node_features[0].shape[1]
    print(f"create a array of shape", nb_graphs, max_nodes, max_nodes, feature_dim)
    graphs_edge_features_from_nodes = jnp.zeros(
        (nb_graphs, max_nodes, max_nodes, feature_dim)
    )
    for k, node_features in enumerate(graps_node_features):
        for i, node_feature in enumerate(node_features):
            graphs_edge_features_from_nodes = graphs_edge_features_from_nodes.at[
                k, i, i, :
            ].set(node_feature)

    if graphs_edge_features == None:

        if edge_featur_init == "ONE_HOT":
            graphs_edge_features = list()
            for k, node_features in enumerate(graps_node_features):
                nb_nodes = node_features.shape[0]
                tmp = jnp.identity(nb_nodes**2)
                tmp = jnp.reshape(tmp, (nb_nodes, nb_nodes, nb_nodes**2))
                graphs_edge_features.append(
                    zero_append(tmp, (max_nodes, max_nodes, max_nodes**2))
                )
                graphs_edge_features = jnp.array(graphs_edge_features)
        if edge_featur_init == "BIAS":
            graphs_edge_features = jnp.full(
                graphs_edge_features_from_nodes.shape[:-1], 1
            )
            graphs_edge_features = jnp.expand_dims(graphs_edge_features, 3)
        else:
            raise Exception("Parameter edge_featur_init must be ONE_HOT or BIAS")

    return jnp.append(graphs_edge_features_from_nodes, graphs_edge_features, axis=3)


def diag(x, batched=True):
    """
    Arange a 2-dim arrary into a 3-dim array.
    Where the 3-dim array has in the channel
    dimension diagonal matricies filled
    with the values from the 2-dim ijnput.
    e.g
    diag(jnp.array[[1,3],[3,4], [5,6]])
    = [ [[1,0], [0,3]], [[3,0], [0,4]], [[5,0], [0,6]]]
    """
    if batched:
        out = jnp.zeros((x.shape[0], x.shape[1], x.shape[1]))
        for i in range(0, x.shape[1]):
            out = out.at[:, i, i].set(x[:, i])
    else:
        out = jnp.zeros((x.shape[0], x.shape[0]))
        out = out.at[jnp.diag_indices(out.shape[0])].set(x)
    return out


def calc_graph_conv_pattern(A, batched=True):
    A_tilde = A + jnp.identity(A.shape[1])
    A_tilde = A_tilde.at[A_tilde == 2].set(1)
    if batched:
        D_tilde = jnp.sum(A_tilde, axis=2)
    else:
        D_tilde = jnp.sum(A_tilde, axis=1)
    D_tilde = 1 / jnp.sqrt(D_tilde)
    D_tilde = diag(D_tilde, batched)
    return D_tilde @ A_tilde @ D_tilde


def expand_pattern_at_channels_dim(pattern_in, nr_channels, batched=True):
    """
    Expand a (batched) two dimensional pattern
    into a three dimensional pattern. The size of the added
    dimension is determined by nr_channels.
    The channe
    """

    if batched:
        out = jnp.zeros(
            (
                pattern_in.shape[0],
                pattern_in.shape[1],
                nr_channels,
                pattern_in.shape[1],
                nr_channels,
            )
        )
        for k in range(pattern_in.shape[0]):
            for i in range(nr_channels):
                out = out.at[k, :, i, :, i].set(pattern_in[k, :])
    else:
        out = jnp.zeros(
            (pattern_in.shape[1], nr_channels, pattern_in.shape[1], nr_channels)
        )
        for i in range(nr_channels):
            out = out.at[:, i, :, i].set(pattern_in)
    return out


def calc_graph_conv_patterns(As):
    """
    calcualte the graph convolution pattern for each graph
    """
    patterns = list()
    for A in As:
        p = calc_graph_conv_pattern(A, False)
        patterns.append(expand_pattern_at_channels_dim(p, 7, False))
    patterns = jnp.array(patterns)
    return patterns


def pattern_preperation(edge_index, nb_graphs, max_nodes, two_wl_radius=[1]):
    """
    As: List,
    """

    max_edges = max([x.shape[1] for x in edge_index])

    # need the adjacency matrix for the 2WL pattern
    As = [to_dense(e, len(e)) for e in edge_index]
    # unify the sizes of all adjacency matricies in the dataset, for the pattern callculation
    As = [zero_append(a, (max_nodes, max_nodes)) for a in As]

    # calculate the graph convolution pattern for each graph (dense pattern)
    # graph_conv_pattern = calc_graph_conv_patterns(As)

    print("Preparing GCN pattern")
    # calculate the graph convolution pattern for each graph (sparse pattern)
    edge_index = jnp.array([zero_append(x, (2, max_edges)) for x in edge_index])
    graph_conv_pattern = jnp.swapaxes(edge_index, 1, 2)
    graph_conv_pattern = jnp.expand_dims(graph_conv_pattern, 2)
    graph_conv_pattern = jnp.array(graph_conv_pattern, dtype="int32")

    print("Preparing 2WL pattern")
    # calculate the 2 wl pattern (or patterns if multiple radia are given)
    As = jnp.array(As)
    two_wl_pattern = []
    for radius in two_wl_radius:
        As = r_power_adjacency_matrix(As, radius)  # does nothing, if r = 1
        As_int = neigbourhood_intersections(As)
        As_pattern = jnp.transpose(jnp.array(jnp.nonzero(As_int)))
        two_wl_pattern.append(As_pattern)

    return graph_conv_pattern, two_wl_pattern


def feature_prepeation(
    graps_node_features, graps_edge_features, edge_featur_init, nb_graphs, max_nodes
):

    print("Preparing edge features")
    # update the edge features. If a graph has no edge features given, use a "one hot encoding" for the edges.
    graps_edge_features = initial_edge_features(
        graps_node_features, graps_edge_features, edge_featur_init, nb_graphs, max_nodes
    )

    print("Preparing node features")
    # from a list of node features to a batched tensor of node features for the graph conv alg.
    graps_node_features = [
        zero_append(ef, (max_nodes, ef.shape[1])) for ef in graps_node_features
    ]
    graps_node_features = jnp.expand_dims(jnp.array(graps_node_features), 3)

    return graps_node_features, graps_edge_features


def data_preperation(
    edge_index,
    graps_node_features,
    graps_edge_features,
    edge_featur_init,
    ys,
    dataset_name,
    base_path,
    two_wl_radius,
):
    """
    As, graps_node_features, ys: Lists
    dataset_name, base_path: Strings
    two_wl_radius: List of ints
    """

    nb_graphs = len(graps_node_features)
    max_nodes = len(max(graps_node_features, key=lambda x: len(x)))

    graps_node_features, graps_edge_features = feature_prepeation(
        graps_node_features, graps_edge_features, edge_featur_init, nb_graphs, max_nodes
    )
    graph_conv_pattern, two_wl_pattern = pattern_preperation(
        edge_index, nb_graphs, max_nodes, two_wl_radius
    )

    jnp.save(base_path + f"/ys", ys)
    jnp.save(base_path + f"/graps_node_features", graps_node_features)
    jnp.save(base_path + f"/graphs_edge_features", graps_edge_features)
    jnp.save(base_path + f"/graph_conv_pattern", graph_conv_pattern)
    for r, p in zip(two_wl_radius, two_wl_pattern):
        jnp.save(base_path + f"/two_wl_pattern_radius_{r}", p)
