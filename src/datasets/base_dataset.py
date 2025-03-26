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
    
    def load_img(self, img_path, numpy=True):
        if numpy:
            img = load_numpy(img_path)
        else: 
            img = load_pil(img_path)
        return img
    
    from typing import List, Dict, Generator, Any

    def iter_genuine(self, user_id):
        """
        Generator function to iterate over genuine signatures for a specific user.
        Args:
            user_id: ID of the user (indexes from 0)
            
        Yields:
            Dictionary containing genuine signature with keys 'img' and 'label'
        """
        for item in self._index[user_id]:
            if item.get("label") == 1:
                yield self.load_img(item.get("path"))

    def iter_forged(self, user_id):
        """
        Generator function to iterate over genuine signatures for a specific user.
        Args:
            user_id: ID of the user (indexes from 0)
            
        Yields:
            Dictionary containing genuine signature with keys 'img' and 'label'
        """
        for item in self._index[user_id]:
            if item.get("label") == 0:
                yield self.load_img(item.get("path"))