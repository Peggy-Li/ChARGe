import os
import copy
import json
import faiss
import numpy as np
from numpy import ndarray
from typing import Any, Union

class LazyList:
    """List with entries that are read from memapped files"""
    def __init__(self, offset_files: Union[str, list[str]]):
        if isinstance(offset_files, str):
            self.files = [offset_files]
        else:
            self.files = offset_files
        self.offsets = [
            np.memmap(f + ".offsets", dtype=np.uint64, mode="r") for f in self.files
        ]
        self.samples = [o.shape[0] for o in self.offsets]
        self.cs = np.cumsum(np.array(self.samples, dtype=np.uint64), dtype=np.uint64)
        self.total_samples = sum(self.samples)
        # Clean memmapped files so that the object can be pickled
        self.offsets = None
        self.non_nan_ind_to_original_ind: dict[int, int] = {}
        self.mapping_to_non_nan: dict[int, int] = {}

    def __getitem__(self, idx):
        if self.mapping_to_non_nan:
            idx = self.mapping_to_non_nan[idx]
        if self.non_nan_ind_to_original_ind:
            idx = self.non_nan_ind_to_original_ind[idx]
        line = self.get_line(idx)
        sample = json.loads(line.strip())
        return sample

    def _lazy_reload(self):
        if self.offsets is not None:
            return

        self.offsets = [
            np.memmap(f + ".offsets", dtype=np.uint64, mode="r") for f in self.files
        ]
        self.files = [np.memmap(f, dtype=np.uint8, mode="r") for f in self.files]

    def __len__(self):
        if self.non_nan_ind_to_original_ind:
            return len(self.non_nan_ind_to_original_ind)
        return self.total_samples

    def get_line(self, idx) -> str:
        """Return line from input file, including any newline"""
        self._lazy_reload()

        # Find file
        f = self.cs.searchsorted(idx, side="right")
        local_idx = idx - (self.cs[f - 1].item() if f > 0 else 0)

        # Find offset
        if local_idx == 0:
            soff = 0
        else:
            soff = self.offsets[f][local_idx - 1].item() + 1
        eoff = self.offsets[f][local_idx].item()

        # Return string
        line = self.files[f][soff:eoff].tobytes().decode("utf-8")
        # line = self.files[f][soff:eoff]
        return line

    def remove_nan(self, not_nan_indices):
        """Make this list behave as if transformed by self.data = [self.data[i] for i in not_nan_pos]
        Example: Starting indices were 0, 1, 2. 1 was nan, leaving not_nan_pos = 0, 2
        self[1] should return 2.
        """
        if self.non_nan_ind_to_original_ind:
            raise ValueError(f'removed nan already called. unclear if these new inds are relative to original data or already cleaned data')
        self.non_nan_ind_to_original_ind = {non_nan_ind: orig_ind for (non_nan_ind, orig_ind) in enumerate(not_nan_indices)}

    def _set_subset_inds(self, inds: list[int], repeatable=False) -> None:
        """Make list a subset of itself with new inds, setting the relevant mapping files
        repeatable: Will make a furthuer subset of itself
        """
        if not repeatable and self.mapping_to_non_nan:
            raise ValueError('_set_subset_inds already called before. Pass in repeatable=True to make a further subset')
        if not self.mapping_to_non_nan:
            self.mapping_to_non_nan = {new_ind: old_ind for (new_ind, old_ind) in enumerate(inds)}
        else:
            # Eg: Original data = 0, 10, 20, 30, 40
            # First subset inds [0, 2, 4], giving subset of [0, 20, 40], mapping_to_non_nan = {0: 0, 1: 2, 2: 4}
            # Second subset inds [0, 2], giving subset of [0, 40], mapping_to_non_nan = {0: 0, 1: 4}
            new_mapping_to_non_nan = {new_ind: self.mapping_to_non_nan[old_ind] for (new_ind, old_ind) in enumerate(inds)}
            self.mapping_to_non_nan = new_mapping_to_non_nan

    def get_subset(self, inds) -> 'LazyList':
        """Return new LazyList that uses a subset of the original"""
        new_lazy_list = copy.copy(self)
        new_lazy_list._set_subset_inds(inds)
        return new_lazy_list



class FaissDataRetriever:
    def __init__(self, data_path: str, emb_path: str, data_format: str = 'json') -> None:
        """
        Args:
            data_path (str): path to data file for retrieval. Must be iterable (e.g., a list)
            emb_path (str): path to npy file containing the embedding vectors for 'data_path'
            data_format (str): data file format for 'data_path' (default: 'json')
        """
        self.data_path = data_path
        self.emb_path = emb_path
        self.data_format = data_format
        
        # Load the data file into an iterable
        match data_format:
            case 'json':
                self.data = self._load_json(data_path)
            case _:
                raise NotImplementedError

        # Load the embedding file and set up the FAISS index
        emb = np.load(emb_path)
        dim = emb.shape[1]
        nan_mask = np.isnan(emb).any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            emb = emb[~nan_mask]
            print(f'{n_nan} NaNs found in embedding file. Removing them, {len(emb)} left...')

            not_nan_pos = np.where(~nan_mask)[0]
            if not isinstance(self.data, LazyList):
                self.data = [self.data[i] for i in not_nan_pos]
            else:
                self.data.remove_nan(not_nan_pos)

        if emb.shape[0] != len(self.data):
            raise ValueError(f'Number of embeddings ({emb.shape[0]}) does not match number of data points ({len(self.data)})')


        # self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.metric_type = faiss.METRIC_Jaccard
        self.faiss_index.add(emb)


    @staticmethod
    def _load_json(filename: str) -> list[dict]:
        offset_file = str(filename) + ".offsets"
        if os.path.exists(offset_file):
            print(f'Using offset file {offset_file}')
            return LazyList(filename)
        with open(filename, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def search_similar(self, query: ndarray, k: int) -> tuple[list[list[float]], list[list[int]], list[list[Any]]]:
        D, I = self.faiss_index.search(query, k)
        similar = []
        for row in I:
            similar.append([self.data[i] for i in row])
        return D.tolist(), I.tolist(), similar
