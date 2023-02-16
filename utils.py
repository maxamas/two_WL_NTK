import neural_tangents as nt
from neural_tangents import stax
from jax import numpy as np
import jax

def neigbourhood_intersections(A):
    """
    Interpretation of out:
    out[a,b,c,d] == 1 if we intersect the nodes
    c and d from batch a, can we reach node b?
    batch x depth x height x wide
    """
    # broadcasting like this:
    # A_full_1 = np.full((A.shape[0], A.shape[1], A.shape[2], A.shape[2]), A)
    # does not work if dim 0 is > 1. Thus loop over the batch dimension

    A_full_1 = np.full((A.shape[1], A.shape[2], A.shape[2]), A[0,:])
    A_full_1 = np.expand_dims(A_full_1, 0)
    for i in range(1, A.shape[0]):
        A_full_1_tmp = np.full((A.shape[1], A.shape[2], A.shape[2]), A[i,:])
        A_full_1_tmp = np.expand_dims(A_full_1_tmp, 0)
        A_full_1 = np.append(A_full_1, A_full_1_tmp, 0)
    # flip stacked A such that the height x wide slices become the depth x wide slices
    A_full_2 = np.swapaxes(A_full_1, 1, 2)
    out = np.array(np.logical_and(A_full_1, A_full_2), dtype="int32")
    return out

def row_wise_karthesian_prod(a, b):
    """
    Returns the row wise cartesian product of 
    two arrays a and b.
    a: a_n x a_m
    b: b_n x b_m
    returns 2 dim array a_n*b_n x a_m + b_m
    example:
    >a = np.array([[1,1,1], [2,2,2]])
    >b = np.array([[1,1,1], [2,2,2], [3,3,3]])*10
    >row_wise_karthesian_prod(a, b)
    > Array([[[ 1,  1,  1, 10, 10, 10],
        [ 2,  2,  2, 10, 10, 10]],

       [[ 1,  1,  1, 20, 20, 20],
        [ 2,  2,  2, 20, 20, 20]],

       [[ 1,  1,  1, 30, 30, 30],
        [ 2,  2,  2, 30, 30, 30]]], dtype=int32)
    """
    a_2 = np.full((b.shape[0], a.shape[0], a.shape[1]), a)
    b_2 = np.full((a.shape[0], b.shape[0], b.shape[1]), b)
    b_3 = np.swapaxes(b_2, 0, 1)
    a_2 = np.reshape(a_2, (-1, a.shape[1]))
    b_3 = np.reshape(b_3, (-1, b.shape[1]))
    out = np.append(a_2, b_3, axis=1)
    return out 

def linear_index(A, ns):
    """
    calculate the edge index of
    an array given by an edge list.
    layed out linearaly.
    A: A edge list of shape [-1, k]
    ns: A list of lenght k giving the size of each dimension
    """
    ns = list(ns)
    ns = ns[1:] + [1]
    out = np.zeros(A.shape[0])
    for i, n in enumerate(ns):
        out += A[:,i] * np.product(np.array(ns[i:]))
    return np.array(out, dtype="int32")

def to_dense(node_list, size):
  """
  Naive implementation, to get a
  adjacency matrix from a node list.
  Node list 2xn -> adjacency matrix nxn
  """
  A = np.zeros((size, size))
  node_list = node_list.tolist()
  for i,j in zip(node_list[0], node_list[1]):
    A = A.at[i,j].set(1)
  return A
  
def r_power_adjacency_matrix(A, r):
  """
  Calculate the r-power for adjacency matrix A.
  From a 3d tensor with batch dimension
  """
  last_As = A
  for i in range(r-1):
    next_As = np.matmul(last_As, A)
    next_As = next_As + last_As
    next_As = next_As.at[next_As != 0].set(1)
    last_As = next_As
  
  return next_As

def zero_append(a, shape):
    """
    Add zero columns and rows to the array 
    a, to make it of shape size x size.
    """
    out = np.zeros((shape[0],shape[1]))
    out = out.at[:a.shape[0],:a.shape[1]].set(a)
    return out
