from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import neural_tangents as nt
from neural_tangents import stax
from jax import numpy as np
import jax
from utils import *

def initial_edge_features(graps_node_features, nb_graphs, max_nodes):
    """
    get the initial edge feature matrix for the 2 WL 
    algorithm.
    when the graph specification has only node features
    as a batch x nodes x nodes x channels array
    """
    feature_dim = graps_node_features[0][0].shape[0]

    graphs_edge_features = np.zeros((nb_graphs, max_nodes, max_nodes, feature_dim))

    for k, node_features in enumerate(graps_node_features):
        for i, node_feature in enumerate(node_features):
            graphs_edge_features = graphs_edge_features.at[k,i,i,:].set(node_feature)

    return graphs_edge_features


def diag(x, batched=True):
  """
  Arange a 2-dim arrary into a 3-dim array.
  Where the 3-dim array has in the channel
  dimension diagonal matricies filled
  with the values from the 2-dim input.
  e.g 
  diag(np.array[[1,3],[3,4], [5,6]])
  = [ [[1,0], [0,3]], [[3,0], [0,4]], [[5,0], [0,6]]]
  """
  if batched:
    out = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(0, x.shape[1]):
      out = out.at[:,i,i].set(x[:,i])
  else:
    out = np.zeros((x.shape[0], x.shape[0]))
    out = out.at[np.diag_indices(out.shape[0])].set(x)
  return out


def calc_graph_conv_pattern(A, batched=True):
  A_tilde = A + np.identity(A.shape[1])
  A_tilde = A_tilde.at[A_tilde == 2].set(1)
  if batched:
    D_tilde = np.sum(A_tilde, axis=2)
  else:
    D_tilde = np.sum(A_tilde, axis=1)
  D_tilde = 1/np.sqrt(D_tilde)
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
      out = np.zeros((pattern_in.shape[0],
                          pattern_in.shape[1], nr_channels, 
                          pattern_in.shape[1], nr_channels))
      for k in range(pattern_in.shape[0]):
        for i in range(nr_channels):
          out = out.at[k,:,i,:,i].set(pattern_in[k,:])
  else:
    out = np.zeros((pattern_in.shape[1], nr_channels, 
                    pattern_in.shape[1], nr_channels))
    for i in range(nr_channels):
      out = out.at[:,i,:,i].set(pattern_in)
  return out

def calc_graph_conv_patterns(As):
    """
    calcualte the graph convolution pattern for each graph
    """
    patterns = list()
    for A in As:
        p = calc_graph_conv_pattern(A, False)
        patterns.append(expand_pattern_at_channels_dim(p, 7, False))
    patterns = np.array(patterns)
    return patterns

def pattern_preperation(As, nb_graphs, max_nodes, two_wl_radius = [1]):
    """
    As: List, 
    """

    # unify the sizes of all adjacency matricies in the dataset, for the pattern callculation
    As = [zero_append(a, (max_nodes, max_nodes)) for a in As]

    # calculate the graph convolution pattern for each graph
    graph_conv_pattern = calc_graph_conv_patterns(As)
    
    # calculate the 2 wl pattern (or patterns if multiple radia are given)
    As = np.array(As)
    two_wl_pattern = []
    for radius in two_wl_radius:
        if radius == 1:
            As_int = neigbourhood_intersections(As)
            As_pattern = np.transpose(np.array(np.nonzero(As_int)))
        two_wl_pattern.append(As_pattern)

    return graph_conv_pattern, two_wl_pattern


def feature_prepeation(graps_node_features, nb_graphs, max_nodes):
    # from node to edge fetures for the 2 wl alg.
    graps_edge_features = initial_edge_features(graps_node_features, nb_graphs, max_nodes)

    # from a list of node features to a batched tensor of node features for the graph conv alg.
    graps_node_features = [zero_append(ef, (max_nodes, ef.shape[1])) for ef in graps_node_features]
    graps_node_features = np.expand_dims(np.array(graps_node_features), 3)

    return graps_node_features, graps_edge_features


def data_preperation(As, graps_node_features, ys, dataset_name, base_path, two_wl_radius):
    """
    As, graps_node_features, ys: Lists
    dataset_name, base_path: Strings
    two_wl_radius: List of ints
    """
    nb_graphs = len(graps_node_features)
    max_nodes = len(max(graps_node_features, key=lambda x: len(x)))

    graps_node_features, graps_edge_features = feature_prepeation(graps_node_features, nb_graphs, max_nodes)
    graph_conv_pattern, two_wl_pattern = pattern_preperation(As, nb_graphs, max_nodes, two_wl_radius)

    np.save(base_path + f"/ys", ys)
    np.save(base_path + f"/graps_node_features", graps_node_features)
    np.save(base_path + f"/graphs_edge_features", graps_edge_features)
    np.save(base_path + f"/graph_conv_pattern", graph_conv_pattern)
    for r, p in zip(two_wl_radius, two_wl_pattern):
        np.save(base_path + f"/two_wl_pattern_radius_{r}", p)





