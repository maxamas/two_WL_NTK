from jax import numpy as jnp
import jax


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
