import os

import pandas as pd
from utils import *
import neural_tangents as nt
from neural_tangents import stax
from jax import numpy as jnp
from jax import random
import jax
from jax import jit, grad, vmap
from jax.example_libraries import optimizers
from jax.nn import log_softmax
from jax._src.typing import Array, Shape
from typing import Callable, Tuple, List
import numpy as np


def accuracy(ys, logits):
    return jnp.mean((logits > 0) == ys)


def cross_entropy(ys, logits):
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return jnp.mean(-ys * log_p - (1 - ys) * log_not_p)


def balance(ys):
    one_indices = jnp.array(jnp.nonzero(ys))[0, :]
    zero_indices = jnp.array(jnp.nonzero(ys == 0))[0, :]

    one_count = int(one_indices.shape[0])
    zero_count = int(zero_indices.shape[0])

    if one_count > zero_count:
        one_indices = one_indices[:zero_count]
    else:
        zero_indices = zero_indices[:one_count]

    indices = jnp.append(one_indices, zero_indices)

    # shuffel the indices
    key = random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    indices = random.permutation(subkey, indices)

    return indices


def val_test_split_pattern_array(
    pattern: Array, train_samples: Array, val_samples: Array, method: str
) -> Tuple[Array, Array]:
    """
    Helper function, because the 2WL pattern and the GCN pattern
    are indexed in a different way. The GCN pattern has batches,
    the 2WL pattern has the patterns of all batches in on 2dim array.
    """
    if method in ["gcn", "GCN"]:
        pattern_val = jnp.take(pattern, val_samples, axis=0)
        pattern_train = jnp.take(pattern, val_samples, axis=0)

        return pattern_train, pattern_val

    elif method in ["2WL", "TWL", "2wl", "twl"]:

        pattern_train = pattern[column_in_values(pattern[:, 0], train_samples), :]
        pattern_val = pattern[column_in_values(pattern[:, 0], val_samples), :]

        return pattern_train, pattern_val

    else:
        raise Exception(
            f"Argument method {method} is unknown in index_pattern_array function call"
        )


def cv_splits(
    Y: Array, nb_folds: int, balance_classes: bool, fill_last_fold: bool = True
):

    # TODO implement the balance_classes option

    dataset_lenght = Y.shape[0]
    indices = jnp.array(range(dataset_lenght), dtype="int32")
    key = random.PRNGKey(1701)
    key, subkey = jax.random.split(key)
    indices = random.permutation(subkey, indices)

    if not dataset_lenght % nb_folds == 0:
        samples_per_fold = dataset_lenght // nb_folds + 1
    else:
        samples_per_fold = dataset_lenght // nb_folds + 1

    out = jnp.array(jnp.zeros(nb_folds * samples_per_fold), dtype="int32")
    out = out.at[:dataset_lenght].set(indices)

    if fill_last_fold:
        key, subkey = jax.random.split(key)
        fill_indices = jax.random.choice(
            subkey, indices, [out.shape[0] - dataset_lenght]
        )
        out = out.at[dataset_lenght:].set(fill_indices)

    out = jnp.reshape(out, (nb_folds, samples_per_fold))
    return out


def cross_validate(
    X: Array,
    Y: Array,
    pattern: Array,
    cv_folds: int,
    init_fn: Callable,
    apply_fn: Callable,
    learning_rate: float,
    epochs: int,
    nn_type: str,
    loss: Callable,
    grad_loss: Callable,
    balance_classes: bool = False,
):

    cv_folds_idxs = cv_splits(Y, cv_folds, balance_classes)

    train_losses_cv_runs = list()
    val_losses_cv_runs = list()
    train_acc_cv_runs = list()
    val_acc_cv_runs = list()

    for cv_hold_out_fold_idx in range(cv_folds):
        print(f"Start CV fold: {cv_hold_out_fold_idx}")

        all_folds = range(cv_folds)
        val_samples = cv_folds_idxs[[cv_hold_out_fold_idx], :]
        train_samples = cv_folds_idxs[
            list(all_folds[:cv_hold_out_fold_idx])
            + list(all_folds[cv_hold_out_fold_idx + 1 :]),
            :,
        ]

        val_samples = jnp.reshape(val_samples, [-1])
        train_samples = jnp.reshape(train_samples, [-1])
        print(
            f"nb train samples: {train_samples.shape[0]} | nb val samples: {val_samples.shape[0]}"
        )

        X_val = jnp.take(X, val_samples, axis=0)
        Y_val = jnp.take(Y, val_samples, axis=0)

        X_train = jnp.take(X, train_samples, axis=0)
        Y_train = jnp.take(Y, train_samples, axis=0)

        pattern_train, pattern_val = val_test_split_pattern_array(
            pattern, train_samples, val_samples, nn_type
        )

        train_losses, val_losses, train_acc, val_acc = train_loop(
            X_train,
            Y_train,
            pattern_train,
            X_val,
            Y_val,
            pattern_val,
            init_fn,
            apply_fn,
            learning_rate,
            epochs,
            loss,
            grad_loss,
        )

        train_losses_cv_runs.append(train_losses)
        val_losses_cv_runs.append(val_losses)
        train_acc_cv_runs.append(train_acc)
        val_acc_cv_runs.append(val_acc)

    return train_losses_cv_runs, val_losses_cv_runs, train_acc_cv_runs, val_acc_cv_runs


def train_loop(
    X_train: Array,
    Y_train: Array,
    pattern_train: Array,
    X_val: Array,
    Y_val: Array,
    pattern_val: Array,
    init_fn: Callable,
    apply_fn: Callable,
    learning_rate: float,
    epochs: int,
    loss: Callable,
    grad_loss: Callable,
) -> Tuple[List[float], List[float], List[float], List[float]]:

    key = random.PRNGKey(1701)
    key, subkey = jax.random.split(key)
    _, params = init_fn(subkey, X_train.shape)

    opt_init, opt_apply, get_params = optimizers.adam(learning_rate)
    opt_apply = jit(opt_apply)
    state = opt_init(params)

    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    for epoch in range(epochs):

        params = get_params(state)
        state = opt_apply(
            epoch, grad_loss(params, X_train, Y_train, pattern_train), state
        )

        params = get_params(state)

        Y_hat_train = apply_fn(params, X_train, pattern=pattern_train)
        Y_hat_val = apply_fn(params, X_val, pattern=pattern_val)

        train_losses += [loss(Y_train, Y_hat_train)]
        val_losses += [loss(Y_val, Y_hat_val)]
        train_acc += [accuracy(Y_train, Y_hat_train)]
        val_acc += [accuracy(Y_val, Y_hat_val)]

        if epoch % 10 == 0:
            print(
                f"\t train loss: {train_losses[-1]:.2f} | val loss: {val_losses[-1]:.2f} | train acc: {train_acc[-1]:.4f} | val acc: {val_acc[-1]:.4f}"
            )

    return train_losses, val_losses, train_acc, val_acc


def save_GD_raw_results(
    dataset,
    training_method,
    nn_type,
    epochs,
    utc_time,
    repo_path,
    cv_folds,
    training_results,
):
    # prepare results and save them
    train_losses_cv_runs_array = np.array([np.array(i) for i in training_results[0]])
    val_losses_cv_runs_array = np.array([np.array(i) for i in training_results[1]])
    train_acc_cv_runs_array = np.array([np.array(i) for i in training_results[2]])
    val_acc_cv_runs_array = np.array([np.array(i) for i in training_results[3]])

    # save the raw data
    results_path = repo_path + f"/Results/{dataset}/{training_method}/{nn_type}"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    np.save(
        results_path
        + f"/{utc_time}_train_losses_CV_folds_{cv_folds}_epochs_{epochs}.npy",
        train_losses_cv_runs_array,
    )
    np.save(
        results_path
        + f"/{utc_time}_val_losses_CV_folds_{cv_folds}_epochs_{epochs}.npy",
        val_losses_cv_runs_array,
    )
    np.save(
        results_path + f"/{utc_time}_train_acc_CV_folds_{cv_folds}_epochs_{epochs}.npy",
        train_acc_cv_runs_array,
    )
    np.save(
        results_path + f"/{utc_time}_val_acc_CV_folds_{cv_folds}_epochs_{epochs}.npy",
        val_acc_cv_runs_array,
    )


def save_core_results(
    val_acc_cv_runs_array,
    utc_time,
    dataset,
    training_method,
    nn_type,
    cv_folds,
    repo_path,
):
    # use for each cv run the epoch with the highest validation acc
    max_acc_for_cv_folds = np.amax(val_acc_cv_runs_array, 1)
    mean = np.mean(max_acc_for_cv_folds)
    std = np.std(max_acc_for_cv_folds)
    min, q_25, q_50, q_75, max = tuple(
        np.quantile(max_acc_for_cv_folds, [0, 0.25, 0.5, 0.75, 1.0])
    )

    print("mean:", mean)
    print("std:", std)
    print("min:", min)
    print("q_25:", q_25)
    print("q_50:", q_50)
    print("q_75:", q_75)
    print("max:", max)

    # append the reults of the run to the results csv
    result_csv_path = repo_path + "/Results/results.csv"
    result_table_append = pd.DataFrame(
        [
            [
                utc_time,
                dataset,
                training_method,
                nn_type,
                cv_folds,
                "accuracy",
                mean,
                std,
                min,
                q_25,
                q_50,
                q_75,
                max,
            ]
        ],
        columns=[
            "UTC Time",
            "Dataset",
            "Training Method",
            "NN Type",
            "Nb CV folds",
            "Metric",
            "mean",
            "std",
            "min",
            "q_25",
            "q_50",
            "q_75",
            "max",
        ],
    )

    if not os.path.exists(result_csv_path):
        result_table_append.to_csv(result_csv_path, index=False)
    else:
        result_table = pd.read_csv(result_csv_path)
        result_table.append(result_table_append).to_csv(result_csv_path, index=False)
