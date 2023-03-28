import copy
import os
from jax import numpy as jnp
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union, List, Dict
import random
import jax
from collections.abc import Generator
from jax._src.typing import Array


class GCN_Obs:

    id: int
    file_path: str
    is_loaded: bool

    def __init__(self, file_path: str, id: int) -> None:
        self.file_path = file_path
        self.id = id
        self.is_loaded = False

    def load_from_file(self) -> None:
        if not self.is_loaded:
            self.edge_list: Array = jnp.load(self.file_path + f"/edge_list.npy")
            self.node_features: Array = jnp.load(self.file_path + f"/node_features.npy")
            self.nb_nodes: int = self.node_features.shape[0]
            self.y: Array = jnp.load(self.file_path + f"/y.npy")
            self.is_loaded = True

    def clear_memory(self) -> None:
        self.is_loaded = False
        del self.edge_list
        del self.node_features
        del self.nb_nodes
        del self.y


class TWL_Obs:

    id: int
    file_path: str
    is_loaded: bool

    def __init__(self, file_path: str, id: int) -> None:
        self.file_path = file_path
        self.id = id
        self.is_loaded = False

    def load_from_file(self) -> None:
        if not self.is_loaded:
            self.ref_matrix: Array = jnp.load(self.file_path + f"/ref_matrix.npy")
            self.edge_features: Array = jnp.load(self.file_path + f"/edge_features.npy")
            self.nb_edges: int = self.edge_features.shape[0]
            self.y: Array = jnp.load(self.file_path + f"/y.npy")
            self.is_loaded = True

    def clear_memory(self) -> None:
        self.is_loaded = False
        del self.ref_matrix
        del self.edge_features
        del self.nb_edges
        del self.y


class Dataloader:

    observations: List
    nb_samples: int

    def __init__(
        self, file_path: str, nb_train_samples: int, only_batch_in_mem: bool = False
    ) -> None:
        self.only_batch_in_mem = only_batch_in_mem

    def train_val_split(self, nb_train_samples: int) -> None:
        ids = list(range(self.nb_samples))
        self.train_ids: List[int] = random.sample(ids, nb_train_samples)
        self.val_ids: List[int] = list(set(ids).difference(set(self.train_ids)))

    def set_train_val_ids(self, train_ids: List[int], val_ids: List[int]) -> None:
        self.train_ids: List[int] = train_ids
        self.val_ids: List[int] = val_ids

    def __load_batch__(self, batch: List[int]) -> None:
        for ob_indx in batch:
            self.observations[ob_indx].load_from_file()

    def __clear_batch__(self, batch: List[int]) -> None:
        for ob_indx in batch:
            self.observations[ob_indx].clear_memory()

    def __move_edge_list_reverence_matrix__(
        self,
        el_rm: Array,
        el_rm_graph_indx: Array,
        node_edge_features_graph_indx: Array,
    ):
        graphs_nb_el_rm = jax.ops.segment_sum(
            jnp.ones(el_rm_graph_indx.shape[0]), el_rm_graph_indx
        )
        graphs_num_features = jax.ops.segment_sum(
            jnp.ones(node_edge_features_graph_indx.shape[0]),
            node_edge_features_graph_indx,
        )

        # shift one back and add 0 as fist elemnt
        tmp = jnp.zeros(graphs_num_features.shape)
        graphs_num_features = tmp.at[1:].set(graphs_num_features[:-1])

        el_rm_move_by = jnp.repeat(
            jnp.array(jnp.cumsum(graphs_num_features), dtype="int32"),
            jnp.array(graphs_nb_el_rm, dtype="int32"),
        )

        out = el_rm + jnp.expand_dims(el_rm_move_by, 1)

        return out

    def get_val_arrays(
        self,
    ) -> Dict:
        pass

    def batch_iterator(
        self, batch_size: int, all_sorted: bool = False
    ) -> Iterable[Dict]:
        pass

    def __split_list__(self, indexes: List[int], nb_splits: int):
        split_size = len(indexes) // nb_splits
        remainder = len(indexes) % nb_splits
        start = 0
        splits = []
        for fold in range(nb_splits):
            end = start + split_size + (fold < remainder)
            splits.append(indexes[start:end])
            start = end
        return splits

    def cross_validation_fold_indices(
        self, nb_folds: int
    ) -> List[Tuple[List[int], List[int]]]:
        indexes_shuffled: List[int] = random.sample(
            range(self.nb_samples), self.nb_samples
        )
        indexes_shuffled_splits = self.__split_list__(indexes_shuffled, nb_folds)

        out = []
        for i, val_set in enumerate(indexes_shuffled_splits):
            train_set = indexes_shuffled_splits[:i] + indexes_shuffled_splits[(i + 1) :]
            train_set = [elem for sublist in train_set for elem in sublist]
            out.append((train_set, val_set))

        return out

    def __make_batches_indexes__(
        self, batch_size: int, all_sorted: bool = False
    ) -> List[List[int]]:

        if all_sorted:
            train_ids_rem = list(range(self.nb_samples))
        else:
            train_ids_rem = copy.deepcopy(self.train_ids)

        nb_batches: int = len(train_ids_rem) // batch_size + 1

        out = list()
        for _ in range(nb_batches):

            if len(train_ids_rem) > batch_size:
                if all_sorted:
                    current_batch_idxs = train_ids_rem[:batch_size]
                else:
                    current_batch_idxs = random.sample(train_ids_rem, batch_size)

                train_ids_rem = sorted(
                    list(set(train_ids_rem).difference(set(current_batch_idxs)))
                )
            else:
                current_batch_idxs = train_ids_rem

            out.append(current_batch_idxs)

        return out


class GCN_Dataloader(Dataloader):
    def __init__(
        self, file_path: str, nb_train_samples: int, only_batch_in_mem: bool = False
    ) -> None:
        super().__init__(file_path, nb_train_samples, only_batch_in_mem)
        self.observations: List[GCN_Obs] = list()
        self.__register_observations__(file_path)
        self.train_val_split(nb_train_samples)

    def __register_observations__(self, file_path: str) -> None:
        id = 0
        for folder in os.listdir(file_path):
            if folder[:7] == "gcn_id_":
                self.observations.append(GCN_Obs(file_path + "/" + folder, id))
                id += 1
        self.nb_samples: int = id

    def __arrays_from_batch__(
        self, batch: List[int]
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:

        node_features = self.observations[batch[0]].node_features
        edge_list = self.observations[batch[0]].edge_list
        node_features_graph_indx = jnp.full(
            self.observations[batch[0]].node_features.shape[0], 0
        )
        edge_list_graph_indx = jnp.full(
            self.observations[batch[0]].edge_list.shape[0], 0
        )

        nb_nodes = jnp.array([self.observations[ob_indx].nb_nodes for ob_indx in batch])

        ys = self.observations[batch[0]].y

        for i, ob_indx in enumerate(batch[1:], start=1):
            node_features = jnp.append(
                node_features, self.observations[ob_indx].node_features, 0
            )
            edge_list = jnp.append(edge_list, self.observations[ob_indx].edge_list, 0)
            node_features_graph_indx = jnp.append(
                node_features_graph_indx,
                jnp.full(self.observations[ob_indx].node_features.shape[0], i),
                0,
            )
            edge_list_graph_indx = jnp.append(
                edge_list_graph_indx,
                jnp.full(self.observations[ob_indx].edge_list.shape[0], i),
                0,
            )
            ys = jnp.append(
                ys,
                self.observations[ob_indx].y,
                0,
            )

        return (
            node_features,
            edge_list,
            node_features_graph_indx,
            edge_list_graph_indx,
            nb_nodes,
            ys,
        )

    def get_val_arrays(
        self,
    ) -> Dict:

        self.__load_batch__(self.val_ids)
        (
            node_features,
            edge_list,
            node_features_graph_indx,
            edge_list_graph_indx,
            nb_nodes,
            ys,
        ) = self.__arrays_from_batch__(self.val_ids)

        if self.only_batch_in_mem:
            self.__clear_batch__(self.val_ids)

        edge_list = self.__move_edge_list_reverence_matrix__(
            edge_list, edge_list_graph_indx, node_features_graph_indx
        )

        return {
            "node_features": node_features,
            "edge_list": edge_list,
            "node_features_graph_indx": node_features_graph_indx,
            "edge_list_graph_indx": edge_list_graph_indx,
            "nb_nodes": nb_nodes,
            "ys": ys,
            "nb_graphs": len(self.val_ids),
        }

    def batch_iterator(
        self, batch_size: int, all_sorted: bool = False
    ) -> Iterable[Dict]:

        batches_indexes = self.__make_batches_indexes__(batch_size, all_sorted)

        for batch_indexes in batches_indexes:
            self.__load_batch__(batch_indexes)
            (
                node_features,
                edge_list,
                node_features_graph_indx,
                edge_list_graph_indx,
                nb_nodes,
                ys,
            ) = self.__arrays_from_batch__(batch_indexes)

            edge_list = self.__move_edge_list_reverence_matrix__(
                edge_list, edge_list_graph_indx, node_features_graph_indx
            )
            if self.only_batch_in_mem:
                self.__clear_batch__(batch_indexes)

            yield {
                "node_features": node_features,
                "edge_list": edge_list,
                "node_features_graph_indx": node_features_graph_indx,
                "edge_list_graph_indx": edge_list_graph_indx,
                "nb_nodes": nb_nodes,
                "ys": ys,
                "nb_graphs": len(batch_indexes),
                "id_high": batch_indexes[
                    -1
                ],  # This makes only sense for all_sorted=True need for the kernel calculation
            }


class TWL_Dataloader(Dataloader):
    def __init__(
        self, file_path: str, nb_train_samples: int, only_batch_in_mem: bool = False
    ) -> None:
        super().__init__(file_path, nb_train_samples, only_batch_in_mem)
        self.observations: List[TWL_Obs] = list()
        self.__register_observations__(file_path)
        self.train_val_split(nb_train_samples)

    def __register_observations__(self, file_path: str) -> None:
        id = 0
        for folder in os.listdir(file_path):
            if folder[:7] == "twl_id_":
                self.observations.append(TWL_Obs(file_path + "/" + folder, id))
                id += 1
        self.nb_samples: int = id

    def __arrays_from_batch__(
        self, batch: List[int]
    ) -> Tuple[Array, Array, Array, Array, Array, Array]:

        edge_features = self.observations[batch[0]].edge_features
        ref_matrix = self.observations[batch[0]].ref_matrix
        edge_features_graph_indx = jnp.full(
            self.observations[batch[0]].edge_features.shape[0], 0
        )
        ref_matrix_graph_indx = jnp.full(
            self.observations[batch[0]].ref_matrix.shape[0], 0
        )
        ys = self.observations[batch[0]].y

        nb_edges = jnp.array([self.observations[ob_indx].nb_edges for ob_indx in batch])

        for i, ob_indx in enumerate(batch[1:], start=1):
            edge_features = jnp.append(
                edge_features, self.observations[ob_indx].edge_features, 0
            )
            ref_matrix = jnp.append(
                ref_matrix, self.observations[ob_indx].ref_matrix, 0
            )
            edge_features_graph_indx = jnp.append(
                edge_features_graph_indx,
                jnp.full(self.observations[ob_indx].edge_features.shape[0], i),
                0,
            )
            ref_matrix_graph_indx = jnp.append(
                ref_matrix_graph_indx,
                jnp.full(self.observations[ob_indx].ref_matrix.shape[0], i),
                0,
            )
            ys = jnp.append(ys, self.observations[ob_indx].y, 0)

        return (
            edge_features,
            ref_matrix,
            edge_features_graph_indx,
            ref_matrix_graph_indx,
            nb_edges,
            ys,
        )

    def get_val_arrays(
        self,
    ) -> Dict:

        self.__load_batch__(self.val_ids)
        (
            edge_features,
            ref_matrix,
            edge_features_graph_indx,
            ref_matrix_graph_indx,
            nb_edges,
            ys,
        ) = self.__arrays_from_batch__(self.val_ids)

        if self.only_batch_in_mem:
            self.__clear_batch__(self.val_ids)

        ref_matrix = self.__move_edge_list_reverence_matrix__(
            ref_matrix, ref_matrix_graph_indx, edge_features_graph_indx
        )

        return {
            "edge_features": edge_features,
            "ref_matrix": ref_matrix,
            "edge_features_graph_indx": edge_features_graph_indx,
            "ref_matrix_graph_indx": ref_matrix_graph_indx,
            "nb_edges": nb_edges,
            "ys": ys,
            "nb_graphs": len(self.val_ids),
        }

    def batch_iterator(
        self, batch_size: int, all_sorted: bool = False
    ) -> Iterable[Dict]:
        batches_indexes = self.__make_batches_indexes__(batch_size, all_sorted)
        for batch_indexes in batches_indexes:
            self.__load_batch__(batch_indexes)
            (
                edge_features,
                ref_matrix,
                edge_features_graph_indx,
                ref_matrix_graph_indx,
                nb_edges,
                ys,
            ) = self.__arrays_from_batch__(batch_indexes)

            ref_matrix = self.__move_edge_list_reverence_matrix__(
                ref_matrix, ref_matrix_graph_indx, edge_features_graph_indx
            )
            if self.only_batch_in_mem:
                self.__clear_batch__(batch_indexes)

            yield {
                "edge_features": edge_features,
                "ref_matrix": ref_matrix,
                "edge_features_graph_indx": edge_features_graph_indx,
                "ref_matrix_graph_indx": ref_matrix_graph_indx,
                "nb_edges": nb_edges,
                "ys": ys,
                "nb_graphs": len(batch_indexes),
                "id_high": batch_indexes[
                    -1
                ],  # This makes only sense for all_sorted=True need for the kernel calculation
            }
