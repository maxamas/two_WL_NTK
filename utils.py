from jax import numpy as jnp
import jax
import os
import re
import numpy as np


def accuracy(ys, logits):
    return jnp.mean((logits > 0) == ys)


def bin_cross_entropy(ys, logits):
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return jnp.mean(-ys * log_p - (1 - ys) * log_not_p)


def cross_entropy_loss(y, logits):
    probs = jax.nn.softmax(logits)
    loss = -jnp.sum(y * jnp.log(probs), axis=1)
    return jnp.mean(loss)


def row_wise_karthesian_prod(a, b):
    """
    Returns the row wise cartesian product of
    two arrays a and b.
    a: a_n x a_m
    b: b_n x b_m
    returns 2 dim array a_n*b_n x a_m + b_m
    example:
    >a = jnp.array([[1,1,1], [2,2,2]])
    >b = jnp.array([[1,1,1], [2,2,2], [3,3,3]])*10
    >row_wise_karthesian_prod(a, b)
    > Array([[[ 1,  1,  1, 10, 10, 10],
        [ 2,  2,  2, 10, 10, 10]],

       [[ 1,  1,  1, 20, 20, 20],
        [ 2,  2,  2, 20, 20, 20]],

       [[ 1,  1,  1, 30, 30, 30],
        [ 2,  2,  2, 30, 30, 30]]], dtype=int32)
    """
    a_2 = jnp.full((b.shape[0], a.shape[0], a.shape[1]), a)
    b_2 = jnp.full((a.shape[0], b.shape[0], b.shape[1]), b)
    b_3 = jnp.swapaxes(b_2, 0, 1)
    a_2 = jnp.reshape(a_2, (-1, a.shape[1]))
    b_3 = jnp.reshape(b_3, (-1, b.shape[1]))
    out = jnp.append(a_2, b_3, axis=1)
    return out


def load_kernelmatrix_from_blocks(kernel_path: str):

    kernel_files = []
    y_files = []
    kernel_files_high_ids = []
    y_files_high_ids = []

    for path in os.listdir(kernel_path):
        if path[:3] == "NTK":
            k_f = path
            k_f_id = [int(i) for i in re.findall(r"\d+", k_f)]
            kernel_files.append(k_f)
            kernel_files_high_ids.append(k_f_id)
        elif path[:2] == "Ys":
            y_f = path
            y_f_id = [int(i) for i in re.findall(r"\d+", y_f)]
            y_files.append(y_f)
            y_files_high_ids.append(y_f_id[0])
        else:
            pass

    ids = [i[0] for i in kernel_files_high_ids]
    max_ids = max(ids)

    kernel_matrix = np.empty((max_ids + 1, max_ids + 1))
    ys = np.empty((max_ids + 1))

    for (h_id_1, h_id_2), kf in zip(kernel_files_high_ids, kernel_files):
        kernel_block = np.squeeze(np.load(kernel_path + "/" + kf))
        l_id_1 = 1 + h_id_1 - kernel_block.shape[0]
        l_id_2 = 1 + h_id_2 - kernel_block.shape[1]

        kernel_matrix[l_id_1 : (h_id_1 + 1), l_id_2 : (h_id_2 + 1)] = kernel_block

    for (h_id), yf in zip(y_files_high_ids, y_files):
        ys_block = np.squeeze(np.load(kernel_path + "/" + yf))
        l_id = 1 + h_id - ys_block.shape[0]

        ys[l_id : (h_id + 1)] = ys_block

    return kernel_matrix, ys


if __name__ == "__main__":
    a = load_kernelmatrix_from_blocks("Data/Kernels/MUTAG/GCN/L_10")
