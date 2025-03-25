import random

from torch.utils.data import Dataset

from src.datasets.loading_utils import load_numpy, load_pil

class BaseDataset(Dataset):

    def __init__(self, index):
        self._index = index

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
    
    from typing import List, Dict, Generator, Any

    def iter_genuine(self, user_id):
        """
        Generator function to iterate over genuine signatures for a specific user.
        Args:
            user_id: ID of the user
            
        Yields:
            Dictionary containing genuine signature with keys 'img' and 'label'
        """
        for item in self._index[user_id]:
            if item["label"] == 1:
                yield item

    def iter_forged(self, user_id):
        """
        Generator function to iterate over genuine signatures for a specific user.
        Args:
            user_id: ID of the user
            
        Yields:
            Dictionary containing genuine signature with keys 'img' and 'label'
        """
        for item in self._index[user_id]:
            if item["label"] == 0:
                yield item