from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from jax import numpy as jnp
import os
from dataloader import GCN_Dataloader, TWL_Dataloader
import config

dataset_name = "PROTEINS"
# dataset_name = "MUTAG"


base_path_preprocessed = config.dataloader_base_path + f"/{dataset_name}/GCN"
d_loader = GCN_Dataloader(file_path=base_path_preprocessed, nb_train_samples=160)

batch_iterator_1 = d_loader.batch_iterator(
    config.kernel_calculation_block_size[dataset_name], True
)

for arrays_1 in batch_iterator_1:
    if arrays_1["id_high"] == 1033:
        print("aa")
    print(arrays_1["id_high"])


# for (train_indx, val_indx) in d_loader.cross_validation_fold_indices(3):
#     d_loader.set_train_val_ids(train_indx, val_indx)
#     for (
#         node_features,
#         edge_list,
#         node_features_graph_indx,
#         edge_list_graph_indx,
#         nb_nodes,
#     ) in d_loader.batch_iterator(10):
#         print(edge_list)
#         (
#             node_features,
#             edge_list,
#             node_features_graph_indx,
#             edge_list_graph_indx,
#             nb_nodes,
#         ) = d_loader.get_val_arrays()
#         print(edge_list)


# base_path_preprocessed = config.dataloader_base_path + f"/{dataset_name}/TWL"
# d_loader = TWL_Dataloader(file_path=base_path_preprocessed, nb_train_samples=160)

# for (train_indx, val_indx) in d_loader.cross_validation_fold_indices(3):
#     d_loader.set_train_val_ids(train_indx, val_indx)
#     for (
#         node_features,
#         edge_list,
#         node_features_graph_indx,
#         edge_list_graph_indx,
#         nb_nodes,
#     ) in d_loader.batch_iterator(10):
#         print(edge_list)
#         (
#             node_features,
#             edge_list,
#             node_features_graph_indx,
#             edge_list_graph_indx,
#             nb_nodes,
#         ) = d_loader.get_val_arrays()
#         print(edge_list)
