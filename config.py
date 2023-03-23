base_path_tu_datasets = "Data/TUData"
dataloader_base_path = "Data/Preprocessed/DataLoader"
kernel_base_path = "Data/Kernels"
dataset_names = [
    "MUTAG",
    "PROTEINS",
    "PTC",
    "NCI1",
    "COLORS-3",
    "IMDB-BINARY",
    "IMDB-MULTI",
]
nn_types = ["TWL", "GCN"]

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
    "PTC": {},
    "NCI1": {},
    "COLORS-3": {},
    "IMDB-BINARY": {},
    "IMDB-MULTI": {},
}

twl_gd_hyperparameters = {
    "MUTAG": {
        "learning_rate": 0.01,
        "epochs": 1,
        "layers": 3,
        "parameterization": "standard",
        "layer_wide": 32,
        "output_layer_wide": 1,
        "batch_size": 64,
        "val_set_portion": 0.2,
        "cv_folds": 2,
    },
    "PROTEINS": {},
    "PTC": {},
    "NCI1": {},
    "COLORS-3": {},
    "IMDB-BINARY": {},
    "IMDB-MULTI": {},
}
