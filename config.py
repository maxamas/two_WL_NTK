base_path_tu_datasets = "Data/TUData"
dataloader_base_path = "Data/Preprocessed/DataLoader"
kernel_base_path = "Data/Kernels"
dataset_names = [
    "MUTAG",
    "PTC_MR",
    "COX2",
    "DHFR",
]
nn_types = ["TWL", "GCN"]

kernel_hyperparameters = {
    "MUTAG": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "PTC_MR": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "COX2": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "DHFR": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
}

gcn_gd_hyperparameters = {
    "MUTAG": {
        "learning_rate": 0.001,
        "epochs": 100,
        "layers": 2,
        "parameterization": "standard",
        "layer_wide": 64,
        "output_layer_wide": 1,
        "batch_size": 32,
        "val_set_portion": 0.2,
        "cv_folds": 10,
        "dense_layers": [32, 16],
    },
}

twl_gd_hyperparameters = {
    "MUTAG": {
        "learning_rate": 0.001,
        "epochs": 200,
        "layers": 10,
        "parameterization": "standard",
        "layer_wide": 64,
        "output_layer_wide": 1,
        "batch_size": 32,
        "val_set_portion": 0.2,
        "cv_folds": 10,
    },
}

kernel_calculation_block_size = {
    "MUTAG": 50,
    "PTC_MR": 50,
    "COX2": 50,
    "DHFR": 50,
}
