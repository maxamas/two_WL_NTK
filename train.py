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


def move_pattern_indices(pattern_sub, cv_folds_idx, graphs_nb_patterns):

    # how many patterns has the subset of graphs
    graphs_nb_patterns_sub = graphs_nb_patterns[
        jnp.isin(jnp.array(range(graphs_nb_patterns.shape[0])), cv_folds_idx)
    ]

    # add 0 first element and drop last element
    tmp = jnp.zeros(graphs_nb_patterns_sub.shape)
    pattern_sub_segment_sum_moved = tmp.at[1:].set(graphs_nb_patterns_sub[:-1])

    # add cumsum of number of patterns to the pattern sub array
    pattern_move_by = jnp.repeat(
        jnp.array(jnp.cumsum(pattern_sub_segment_sum_moved), dtype="int32"),
        jnp.array(graphs_nb_patterns_sub, dtype="int32"),
    )

    # move the pattern indices
    pattern_sub_moved = pattern_sub + jnp.expand_dims(pattern_move_by, 1)

    return pattern_sub_moved


def get_train_val_indexes(hold_out_idx, cv_folds: int, cv_folds_idxs, sorted=True):
    all_folds = range(cv_folds)
    val_indxs = jnp.reshape(cv_folds_idxs[[hold_out_idx], :], -1)
    train_indxs = jnp.reshape(
        cv_folds_idxs[
            list(all_folds[:hold_out_idx]) + list(all_folds[hold_out_idx + 1 :]),
            :,
        ],
        -1,
    )

    # this needs to be sortet, because we need to index the edge features in order
    train_indxs = jnp.sort(train_indxs)
    val_indxs = jnp.sort(val_indxs)

    return train_indxs, val_indxs


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


def prepare_data_subset(
    n_type: str,
    graph_indx_full: Array,
    graph_indx_sub: Array,
    pattern_graph_indx_full: Array,
    pattern_segment_sum_full: Array,
    X_full: Array,
    Y_full: Array,
    pattern_full: Array,
):
    # pattern subset
    pattern_sub = move_pattern_indices(
        pattern_full[jnp.isin(pattern_graph_indx_full, graph_indx_sub), :],
        graph_indx_sub,
        pattern_segment_sum_full,
    )

    if n_type == "gcn":
        # node features subsets
        X_sub = X_full[:, jnp.isin(graph_indx_full, graph_indx_sub), :]

        # pattern needs to be 3 dimensional batched for gsn
        pattern_sub = np.expand_dims(pattern_sub, (0, 2))
        pattern_sub = np.append(pattern_sub, pattern_sub, axis=2)
    else:
        # edge features subsets
        X_sub = X_full[jnp.isin(graph_indx_full, graph_indx_sub), :]

    # graph index subsets
    graph_indx_sub = graph_indx_full[jnp.isin(graph_indx_full, graph_indx_sub)]

    # Y subsets
    Y_sub = Y_full[graph_indx_sub]

    nb_graphs_sub = jnp.unique(graph_indx_sub).shape[0]

    return X_sub, Y_sub, pattern_sub, graph_indx_sub, nb_graphs_sub


def cross_validate(
    n_type: "str",
    X: Array,
    Y: Array,
    graph_indx: Array,
    pattern: Array,
    pattern_graph_indx: Array,
    cv_folds: int,
    init_fn: Callable,
    apply_fn: Callable,
    learning_rate: float,
    epochs: int,
    loss: Callable,
    grad_loss: Callable,
    balance_classes: bool = False,
):
    # get an cv_folds x -1 array of indexes to create folds
    cv_folds_idxs = cv_splits(Y, cv_folds, balance_classes)

    # how many patterns has each graph
    pattern_segment_sum = jax.ops.segment_sum(
        jnp.ones(pattern_graph_indx.shape), pattern_graph_indx
    )

    train_losses_cv_runs = list()
    val_losses_cv_runs = list()
    train_acc_cv_runs = list()
    val_acc_cv_runs = list()

    for cv_hold_out_fold_idx in range(cv_folds):
        print(f"Start CV fold: {cv_hold_out_fold_idx}")

        # graph indices which are in training and validation folds
        train_indxs, val_indxs = get_train_val_indexes(
            cv_hold_out_fold_idx, cv_folds, cv_folds_idxs
        )

        (
            train_X,
            Y_train,
            pattern_train,
            graph_indx_train,
            nb_graphs_train,
        ) = prepare_data_subset(
            n_type,
            graph_indx,
            train_indxs,
            pattern_graph_indx,
            pattern_segment_sum,
            X,
            Y,
            pattern,
        )

        (
            val_X,
            Y_val,
            pattern_val,
            graph_indx_val,
            nb_graphs_val,
        ) = prepare_data_subset(
            n_type,
            graph_indx,
            val_indxs,
            pattern_graph_indx,
            pattern_segment_sum,
            X,
            Y,
            pattern,
        )

        train_losses, val_losses, train_acc, val_acc = train_loop(
            train_X,
            graph_indx_train,
            Y_train,
            pattern_train,
            nb_graphs_train,
            val_X,
            graph_indx_val,
            Y_val,
            pattern_val,
            nb_graphs_val,
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
    graph_indx_train: Array,
    Y_train: Array,
    pattern_train: Array,
    nb_graphs_train: Array,
    X_val: Array,
    graph_indx_val: Array,
    Y_val: Array,
    pattern_val: Array,
    nb_graphs_val: Array,
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
            epoch,
            grad_loss(
                params,
                X_train,
                Y_train,
                pattern=pattern_train,
                nb_graphs=nb_graphs_train,
                graph_indx=graph_indx_train,
            ),
            state,
        )

        params = get_params(state)

        Y_hat_train = apply_fn(
            params,
            X_train,
            pattern=pattern_train,
            graph_indx=graph_indx_train,
            nb_graphs=nb_graphs_train,
        )
        Y_hat_val = apply_fn(
            params,
            X_val,
            pattern=pattern_val,
            graph_indx=graph_indx_val,
            nb_graphs=nb_graphs_val,
        )

        Y_hat_train = Y_hat_train[: Y_train.shape[0], :]
        Y_hat_val = Y_hat_train[: Y_val.shape[0], :]

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
