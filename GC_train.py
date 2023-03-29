import os
from jax import numpy as jnp
from dataloader import GCN_Dataloader, TWL_Dataloader, Dataloader
import config
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union, List
import jax
from jax._src.typing import Array
from network_config import get_2wl_network_configuration, get_gcn_network_configuration
import os
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
from utils import bin_cross_entropy, cross_entropy_loss
import pandas as pd


def train_loop(
    get_batch_iterator: Callable,
    input_shape: Tuple,
    val_arrays: Dict,
    init_fn: Callable,
    apply_fn: Callable,
    learning_rate: float,
    epochs: int,
    loss: Callable,
    grad_loss: Callable,
) -> Dict:

    key = random.PRNGKey(1701)
    key, subkey = jax.random.split(key)
    _, params = init_fn(subkey, input_shape)

    opt_init, opt_apply, get_params = optimizers.adam(learning_rate)
    opt_apply = jit(opt_apply)
    state = opt_init(params)

    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []

    for epoch in range(epochs):
        for train_arrays in get_batch_iterator():

            # make update for the batch
            params = get_params(state)
            state = opt_apply(
                epoch,
                grad_loss(params, train_arrays),
                state,
            )

            params = get_params(state)

            Y_hat_train = apply_fn(params, train_arrays)[
                : train_arrays["ys"].shape[0], :
            ]
            train_losses += [loss(train_arrays["ys"], Y_hat_train)]
            train_acc += [accuracy(train_arrays["ys"], Y_hat_train)]

        Y_hat_val = apply_fn(params, val_arrays)[: val_arrays["ys"].shape[0], :]
        val_losses += [loss(val_arrays["ys"], Y_hat_val)]
        val_acc += [accuracy(val_arrays["ys"], Y_hat_val)]

        if epoch % 5 == 0:
            print(
                f"\t train loss: {train_losses[-1]:.2f} | val loss: {val_losses[-1]:.2f} | train acc: {train_acc[-1]:.4f} | val acc: {val_acc[-1]:.4f}"
            )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "params": params,
    }


def gcn_apply_function(apply_fn: Callable) -> Callable:
    return lambda params, x: apply_fn(
        params,
        inputs=x["node_features"],
        pattern=x["edge_list"],
        graph_indx=x["node_features_graph_indx"],
    )


def twl_apply_function(apply_fn: Callable) -> Callable:
    return lambda params, x: apply_fn(
        params,
        inputs=x["edge_features"],
        pattern=x["ref_matrix"],
        graph_indx=x["edge_features_graph_indx"],
    )


def cross_validate(
    folds: int,
    input_shape: Tuple,
    get_apply_function: Callable,
    get_netwok_conf: Callable,
    data_loader: Dataloader,
    hyper_parameter: dict,
    loss: Callable,
) -> Dict:

    init_fn, apply_fn, _ = get_netwok_conf(
        hyper_parameter["layers"],
        hyper_parameter["parameterization"],
        hyper_parameter["layer_wide"],
        hyper_parameter["output_layer_wide"],
    )
    decorated_apply_fn = get_apply_function(apply_fn)

    grad_loss = jit(
        grad(
            lambda params, arrays: loss(
                arrays["ys"],
                (decorated_apply_fn(params, arrays))[: arrays["ys"].shape[0], :],
            )
        )
    )

    train_losses_cv_runs = list()
    val_losses_cv_runs = list()
    train_acc_cv_runs = list()
    val_acc_cv_runs = list()

    for (train_indx, val_indx) in data_loader.cross_validation_fold_indices(folds):
        data_loader.set_train_val_ids(train_indx, val_indx)

        val_arrays = data_loader.get_val_arrays()

        train_res = train_loop(
            lambda: data_loader.batch_iterator(hyper_parameter["batch_size"]),
            input_shape,
            val_arrays,
            init_fn,
            decorated_apply_fn,
            learning_rate=hyper_parameter["learning_rate"],
            epochs=hyper_parameter["epochs"],
            loss=loss,
            grad_loss=grad_loss,
        )

        train_losses_cv_runs.append(train_res["train_losses"])
        val_losses_cv_runs.append(train_res["val_losses"])
        train_acc_cv_runs.append(train_res["train_acc"])
        val_acc_cv_runs.append(train_res["val_acc"])

    return {
        "train_losses_cv_runs": train_losses_cv_runs,
        "val_losses_cv_runs": val_losses_cv_runs,
        "train_acc_cv_runs": train_acc_cv_runs,
        "val_acc_cv_runs": val_acc_cv_runs,
    }


def run_cv(datasets_names: List[str], nn_types: List[str]):
    for dataset_name in datasets_names:
        for nn_type in nn_types:

            print(f"Train {nn_type} on {dataset_name}. ")

            base_path_preprocessed = (
                config.dataloader_base_path + f"/{dataset_name}/{nn_type}"
            )

            if nn_type == "GCN":
                data_loader = GCN_Dataloader(
                    file_path=base_path_preprocessed, nb_train_samples=160
                )
                get_netwok_conf = get_gcn_network_configuration
                get_decorated_apply_fn = gcn_apply_function
                hyper_parameter = config.gcn_gd_hyperparameters[dataset_name]
                input_shape = data_loader.get_val_arrays()["node_features"].shape
            elif nn_type == "TWL":
                data_loader = TWL_Dataloader(
                    file_path=base_path_preprocessed, nb_train_samples=160
                )
                # layer with must be specified as int, but is ignored
                get_netwok_conf = get_2wl_network_configuration
                get_decorated_apply_fn = twl_apply_function
                hyper_parameter = config.twl_gd_hyperparameters[dataset_name]
                input_shape = data_loader.get_val_arrays()["edge_features"].shape
            else:
                print(f"No path for nn_type {nn_type}")
                exit()

            # loss function
            loss = jit(bin_cross_entropy)

            cv_results = cross_validate(
                hyper_parameter["cv_folds"],
                input_shape,
                get_decorated_apply_fn,
                get_netwok_conf,
                data_loader,
                hyper_parameter,
                loss,
            )

            results_path = f"Results/{dataset_name}/Gradien_Descent/{nn_type}"

            time = pd.Timestamp.utcnow().timestamp()

            save_GD_raw_results(
                results_path,
                hyper_parameter,
                time,
                cv_results,
            )

            save_core_results(
                cv_results,
                time,
                dataset_name,
                "Gradient Descent",
                nn_type,
                hyper_parameter["cv_folds"],
            )


def save_GD_raw_results(
    results_path,
    hyper_parameter,
    utc_time,
    training_results,
):
    # prepare results and save them
    train_losses_cv_runs_array = np.array(
        [np.array(i) for i in training_results["train_losses_cv_runs"]]
    )
    val_losses_cv_runs_array = np.array(
        [np.array(i) for i in training_results["val_losses_cv_runs"]]
    )
    train_acc_cv_runs_array = np.array(
        [np.array(i) for i in training_results["train_acc_cv_runs"]]
    )
    val_acc_cv_runs_array = np.array(
        [np.array(i) for i in training_results["val_acc_cv_runs"]]
    )

    epochs = hyper_parameter["epochs"]
    cv_folds = hyper_parameter["cv_folds"]

    # save the raw data
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
    training_results,
    utc_time,
    dataset,
    training_method,
    nn_type,
    cv_folds,
):
    # use for each cv run the epoch with the highest validation acc
    val_acc_cv_runs_array = np.array(
        [np.array(i) for i in training_results["val_acc_cv_runs"]]
    )
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
    result_csv_path = "Results/results.csv"
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


def train(
    dataset_name,
    nn_type,
    base_path_preprocessed=config.dataloader_base_path,
    gcn_hyper_params=config.gcn_gd_hyperparameters,
    twl_hyper_params=config.twl_gd_hyperparameters,
):

    print(f"Train {nn_type} on {dataset_name}!")
    base_path_preprocessed = base_path_preprocessed + f"/{dataset_name}/{nn_type}/"
    if nn_type == "GCN":
        data_loader = GCN_Dataloader(
            file_path=base_path_preprocessed, nb_train_samples=160
        )
        hyper_parameter = gcn_hyper_params[dataset_name]
        input_shape = data_loader.get_val_arrays()["node_features"].shape

        # load the network configuration
        init_fn, apply_fn, _ = get_gcn_network_configuration(
            hyper_parameter["layers"], "standard", hyper_parameter["layer_wide"], 1
        )
        decorated_apply_fn = gcn_apply_function(apply_fn)

    elif nn_type == "TWL":
        data_loader = TWL_Dataloader(
            file_path=base_path_preprocessed, nb_train_samples=160
        )
        hyper_parameter = twl_hyper_params[dataset_name]
        input_shape = data_loader.get_val_arrays()["edge_features"].shape

        # load the network configuration
        init_fn, apply_fn, _ = get_2wl_network_configuration(
            hyper_parameter["layers"], "standard", hyper_parameter["layer_wide"], 1
        )
        decorated_apply_fn = twl_apply_function(apply_fn)
    else:
        print(f"No path for nn_type {nn_type}")
        exit()

    get_batch_iterator = lambda: data_loader.batch_iterator(64)
    val_arrays = data_loader.get_val_arrays()

    # loss function
    loss = jit(bin_cross_entropy)
    grad_loss = jit(
        grad(
            lambda params, arrays: loss(
                arrays["ys"],
                (decorated_apply_fn(params, arrays))[: arrays["ys"].shape[0], :],
            )
        )
    )

    res = train_loop(
        get_batch_iterator,
        val_arrays["edge_features"].shape,
        val_arrays,
        init_fn,
        decorated_apply_fn,
        learning_rate=hyper_parameter["learning_rate"],
        epochs=hyper_parameter["epochs"],
        loss=loss,
        grad_loss=grad_loss,
    )

    print(decorated_apply_fn(res["params"], val_arrays)[: val_arrays["ys"].shape[0], :])


if __name__ == "__main__":

    # # dataset_names = config.dataset_names[:1]
    # # nn_types = config.nn_types[:1]

    # dataset_names = [
    #     # "MUTAG",
    #     # "PROTEINS",
    #     "PTC_MR",
    #     # "NCI1",
    #     # "COLORS-3",  # has no node and edge features
    #     # "IMDB-BINARY",  # has no node and edge features
    #     # "IMDB-MULTI",  # has no node and edge features
    # ]
    # nn_types = [
    #     "TWL",
    #     # "GCN"
    # ]

    # run_cv(dataset_names, nn_types)

    train(
        "MUTAG",
        "TWL",
        base_path_preprocessed=config.dataloader_base_path,
        gcn_hyper_params=config.gcn_gd_hyperparameters,
        twl_hyper_params=config.twl_gd_hyperparameters,
    )
