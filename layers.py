from typing import Callable, Iterable, Optional, Sequence, Tuple, Union
from neural_tangents import Kernel
from neural_tangents._src.stax.requirements import Bool, Diagonal, get_diagonal_outer_prods, layer, mean_and_var, requires, supports_masking
from jax import numpy as np
import jax
from utils import row_wise_karthesian_prod

# pattern is a two dimensional array of shape k x 4
# a edge list from the 4 dimensional intersected adjacency matrix
# R = np.transpose(np.array(np.nonzero(As_int)))
# As_int is four dimensional with batch x #nodes x #nodes x #nodes
# e.g. As_int[a,b,c,:] gives all nodes, which are neighboors
# of nodes b and c in batch a.

# Thus e.g a row from the pattern array [A,B,C,D] tells us,
# that node D is a neighboor of nodes B and C in batch A.

@layer
@supports_masking(remask_kernel=False)
def two_wl_aggregation(n_nodes):
    """
    Return a layer, that implements the gatter
    and scatter operations given the reference Matrix 
    R.
    R is of shape (#graps in the batch *
    # edges of each grap) x 2
    """
    
    init_fn = lambda rng, input_shape: (input_shape, ())
    
    def apply_fn(params,
               inputs: np.ndarray,
               *,
               pattern: Optional[np.ndarray] = None,
               **kwargs):
        num_segments = inputs.shape[0] * n_nodes**2
        
        # edges from v_i to v_j
        # pattern := [A,B,C,D]
        # A = batch, B = node i, C = node j
        # linear_index([A,B,C]) => if we have a batched adjacency matrix
        # what is the index of nodes B, C in batch A, if we index the 
        # array 3 dimensional array linear (c-style)
        e_ij = linear_index(pattern[:,[0,1,2]], inputs.shape)
        # edges from v_i to v_l
        e_il = linear_index(pattern[:,[0,1,3]], inputs.shape)
        # edges from v_l to v_j
        e_lj = linear_index(pattern[:,[0,3,2]], inputs.shape)

        graphs_edge_features = np.reshape(inputs, (-1, inputs.shape[3]))
        x_gamma_1 = np.take(graphs_edge_features, e_il, axis=0)
        x_gamma_2 = np.take(graphs_edge_features, e_lj, axis=0)
        X_gamma_1_sum = jax.ops.segment_sum(x_gamma_1, e_ij, num_segments)
        X_gamma_2_sum = jax.ops.segment_sum(x_gamma_2, e_ij, num_segments)
        X_gamma_sum = np.append(np.expand_dims(X_gamma_1_sum, 1), np.expand_dims(X_gamma_2_sum, 1), 1)
        X_gamma_sum = np.sum(X_gamma_sum, 1)

        out = np.reshape(X_gamma_sum, inputs.shape)
        return out


    def kernel_fn(k: Kernel,
                *,
                pattern: Tuple[Optional[np.ndarray],
                               Optional[np.ndarray]] = (None, None),
                **kwargs):
        
        num_segments = int(np.prod(np.array(k.ntk.shape)))
        
        # arrange the incoming kernel matrix as a flatt array
        # ntk_linear = np.reshape(k.ntk, (-1, 1))
        # we dont need to reshape, as take works also on the multidimensional array

        # a double sum is one sum over the karthesian product of 
        # the sets the two sums sum.
        # pattern has columns: batch, node i, node j, node a
        # the karthesian product (A x B) ist than: 
        # b_A, i_A, j_A, a_A, b_B, i_B, j_B, a_B
        #   0,   1,   2,   3,   4,   5,   6,   7
        # rearrange the columns to: 
        # b_A, b_B, i_A, i_B, j_A, j_B, a_A, a_B
        #   0,   1,   2,   3,   4,   5,   6,   7
        patterns = row_wise_karthesian_prod(pattern[0], pattern[1])       
        patterns = patterns[:,[0,4,1,5,2,6,3,7]]
        # "Interpretation" of the kartesian product:
        # A row from the pattern array [A,B,C,D] tells us,
        # that node D is a neighboor of nodes B and C in batch A.
        # Lets consider only the rows, where columns A,B,C are equal (node e_ij)
        # column D is then the "set" of nodes the "neigborhood" aggregation
        # sum for e_ij sums.
        # Also, consider the pathern from the other graph with 
        # columns [Ab,Bb,Cb,Db]. When take only the rows where 
        # columns Ab, Bb, Cb are equal. Column Db makes then the 
        # indexes to sum over for the "neigborhood" aggregation.
        # To create the double sum we need the 
        # karthesian product of the indexes in column D and 
        # column Db.

        # note: ib <=> i bar
        e_i_j_ib_jb = linear_index(patterns[:,[0,1,2,4,3,5]], k.ntk.shape)
        e_i_a_ib_ab = linear_index(patterns[:,[0,1,2,6,3,7]], k.ntk.shape)
        e_i_a_ab_jb = linear_index(patterns[:,[0,1,2,6,7,5]], k.ntk.shape)
        e_a_j_ib_ab = linear_index(patterns[:,[0,1,6,4,3,7]], k.ntk.shape)
        e_a_j_ab_jb = linear_index(patterns[:,[0,1,6,4,7,5]], k.ntk.shape)

        theta_i_a_ib_ab = jax.ops.segment_sum(np.take(k.ntk, e_i_a_ib_ab), e_i_j_ib_jb, num_segments)
        theta_i_a_ab_jb = jax.ops.segment_sum(np.take(k.ntk, e_i_a_ab_jb), e_i_j_ib_jb, num_segments)
        theta_a_j_ib_ab = jax.ops.segment_sum(np.take(k.ntk, e_a_j_ib_ab), e_i_j_ib_jb, num_segments)
        theta_a_j_ab_jb = jax.ops.segment_sum(np.take(k.ntk, e_a_j_ab_jb), e_i_j_ib_jb, num_segments)

        thetas_linear = np.zeros((theta_i_a_ab_jb.shape[0],1))
        thetas_linear = np.append(thetas_linear, np.expand_dims(theta_i_a_ib_ab, 1), 1)
        thetas_linear = np.append(thetas_linear, np.expand_dims(theta_i_a_ab_jb, 1), 1)
        thetas_linear = np.append(thetas_linear, np.expand_dims(theta_a_j_ib_ab, 1), 1)
        thetas_linear = np.append(thetas_linear, np.expand_dims(theta_a_j_ab_jb, 1), 1)

        theta_linear = np.sum(thetas_linear, 1)
        theta = np.reshape(theta_linear, k.ntk.shape)
        thera = theta + k.ntk
        
        return k.replace(ntk=theta,
                     is_gaussian=True,
                     is_input=False)
    
    return init_fn, apply_fn, kernel_fn