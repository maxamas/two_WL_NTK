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
kernel_calculation_block_size = {
    "MUTAG": 50,
    "PTC_MR": 50,
    "COX2": 50,
    "DHFR": 50,
}
gd_hyperparameters = {
    "MUTAG": {
        "learning_rate": {"TWL": 0.001, "GCN": 0.001},
        "epochs": {"TWL": 100, "GCN": 100},
        "layers": {"TWL": 10, "GCN": 10},
        "parameterization": {"TWL": "standard", "GCN": "standard"},
        "layer_wide": {"TWL": 64, "GCN": 64},
        "output_layer_wide": {"TWL": 1, "GCN": 1},
        "batch_size": {"TWL": 32, "GCN": 32},
        "val_set_portion": {"TWL": 0.2, "GCN": 0.2},
        "cv_folds": {"TWL": 10, "GCN": 10},
    },
    "PTC_MR": {
        "learning_rate": {"TWL": 0.001, "GCN": 0.001},
        "epochs": {"TWL": 100, "GCN": 100},
        "layers": {"TWL": 10, "GCN": 10},
        "parameterization": {"TWL": "standard", "GCN": "standard"},
        "layer_wide": {"TWL": 64, "GCN": 64},
        "output_layer_wide": {"TWL": 1, "GCN": 1},
        "batch_size": {"TWL": 32, "GCN": 32},
        "val_set_portion": {"TWL": 0.2, "GCN": 0.2},
        "cv_folds": {"TWL": 10, "GCN": 10},
    },
    "COX2": {
        "learning_rate": {"TWL": 0.001, "GCN": 0.001},
        "epochs": {"TWL": 100, "GCN": 100},
        "layers": {"TWL": 10, "GCN": 10},
        "parameterization": {"TWL": "standard", "GCN": "standard"},
        "layer_wide": {"TWL": 64, "GCN": 64},
        "output_layer_wide": {"TWL": 1, "GCN": 1},
        "batch_size": {"TWL": 32, "GCN": 32},
        "val_set_portion": {"TWL": 0.2, "GCN": 0.2},
        "cv_folds": {"TWL": 10, "GCN": 10},
    },
    "DHFR": {
        "learning_rate": {"TWL": 0.001, "GCN": 0.001},
        "epochs": {"TWL": 100, "GCN": 100},
        "layers": {"TWL": 10, "GCN": 10},
        "parameterization": {"TWL": "standard", "GCN": "standard"},
        "layer_wide": {"TWL": 64, "GCN": 64},
        "output_layer_wide": {"TWL": 1, "GCN": 1},
        "batch_size": {"TWL": 32, "GCN": 32},
        "val_set_portion": {"TWL": 0.2, "GCN": 0.2},
        "cv_folds": {"TWL": 10, "GCN": 10},
    },
}
