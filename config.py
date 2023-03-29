base_path_tu_datasets = "Data/TUData"
dataloader_base_path = "Data/Preprocessed/DataLoader"
kernel_base_path = "Data/Kernels"
dataset_names = [
    "MUTAG",
    "PROTEINS",
    "PTC_MR",
    "NCI1",
    "COLORS-3",  # has no node and edge features
    "IMDB-BINARY",  # has no node and edge features
    "IMDB-MULTI",  # has no node and edge features
]
nn_types = ["TWL", "GCN"]

kernel_hyperparameters = {
    "MUTAG": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "PROTEINS": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "PTC_MR": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "NCI1": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "COLORS-3": {"parameterization": "ntk", "output_layer_wide": 11, "nb_layers": 10},
    "IMDB-BINARY": {"parameterization": "ntk", "output_layer_wide": 1, "nb_layers": 10},
    "IMDB-MULTI": {"parameterization": "ntk", "output_layer_wide": 3, "nb_layers": 10},
}


gcn_gd_hyperparameters = {
    "MUTAG": {
        "learning_rate": 0.01,
        "epochs": 100,
        "layers": 3,
        "parameterization": "standard",
        "layer_wide": 32,
        "output_layer_wide": 1,
        "batch_size": 64,
        "val_set_portion": 0.2,
        "cv_folds": 10,
    },
    "PROTEINS": {},
    "PTC_MR": {},
    "NCI1": {},
    "COLORS-3": {},
    "IMDB-BINARY": {},
    "IMDB-MULTI": {},
}

twl_gd_hyperparameters = {
    "MUTAG": {
        "learning_rate": 0.01,
        "epochs": 200,
        "layers": 3,
        "parameterization": "standard",
        "layer_wide": 32,
        "output_layer_wide": 1,
        "batch_size": 64,
        "val_set_portion": 0.2,
        "cv_folds": 10,
    },
    "PROTEINS": {},
    "PTC_MR": {},
    "NCI1": {},
    "COLORS-3": {},
    "IMDB-BINARY": {},
    "IMDB-MULTI": {},
}


kernel_calculation_block_size = {
    "MUTAG": 50,
    "PROTEINS": 10,
    "PTC_MR": 50,
    "NCI1": {},
    "COLORS-3": {},
    "IMDB-BINARY": 10,
    "IMDB-MULTI": 50,
}
