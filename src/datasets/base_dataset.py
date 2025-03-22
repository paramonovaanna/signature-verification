import random

from torch.utils.data import Dataset

from src.datasets.loading_utils import load_numpy, load_pil

class BaseDataset(Dataset):

    def __init__(self, index, limit=None, shuffle_limit=False):
        self._index = index
        if limit is not None:
            self._index = self._limit_index(limit, shuffle_limit)

    def _limit_index(self, limit, shuffle_limit):
        if shuffle_limit:
            random.seed(42)
            random.shuffle(self._index)

        index = self._index[:limit]
        return index
    
    def __getitem__(self, ind):
        return self._index[ind]
    
    def __len__(self):
        return len(self._index)
    
    def load_img(self, ind, numpy=False):
        data_dict = self._index[ind]
        label = data_dict["label"]

        img_path = data_dict["path"]
        if numpy:
            img = load_numpy(img_path)
        else: 
            img = load_pil(img_path)
        return img, label

