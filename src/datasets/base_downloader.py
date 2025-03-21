import random
from src.datasets.basedataset import BaseDataset

from PIL import Image

import numpy as np

from tqdm import tqdm 

from skimage.io import imread
from skimage import img_as_ubyte

class BaseDownloader:

    def __init__(self, index, limit=None, shuffle_limit=False):
        self._index = index
        if limit is not None:
            self._index = self._limit_index(limit, shuffle_limit)
        self.data = [] # needs to be loaded by calling load()

    def _limit_index(self, limit, shuffle_limit):
        if shuffle_limit:
            random.seed(42)
            random.shuffle(self.data)

        index = self._index[:limit]
        return index

    def load(self, load_numpy):
        if load_numpy:
            return self.loading_numpy()
        return self.loading_pil()
        
    def loading_numpy(self):
        data = []
        print("Loading images in numpy style...")
        for item in tqdm(self._index):
            img = self.load_img_numpy(item["path"])
            data.append({"img": img, "label": item["label"]})
        self.data = data

    def loading_pil(self):
        data = []
        print("Loading PIL images...")
        for item in tqdm(self._index):
            img = self.load_img_pil(item["path"])
            data.append({"img": img, "label": item["label"]})
        self.data = data

    def load_img_numpy(self, path):
        """
        Load img from disk.

        Args:
            path (str): path to the object.
        Returns:
            img (np.ndarray):
        """
        img = imread(path, as_gray=True)
        return img_as_ubyte(img)
    
    def load_img_pil(self, path):
        """
        Load img from disk.

        Args:
            path (str): path to the object.
        Returns:
            img (Tensor):
        """
        img = Image.open(path).convert("L")
        return img

    def _random_split(self, split, shuffle_split):
        assert 0 <= split <= 1, ("Split should be between 0 and 1")

        train_size = int(len(self.data) * split)

        if shuffle_split:
            random.seed(42)
            random.shuffle(self.data)
        return self.data[:train_size], self.data[train_size:]
    
    def get_test_data(self, instance_transforms=None):
        '''
        Returns a dataset object for testing
        '''
        return {"test": BaseDataset(self.data, instance_transforms["test"])}
    
    def get_partitions(self, split, shuffle_split, instance_transforms=None):
        '''
        Returns a partitions dict object
        '''
        train_data, test_data = self._random_split(split, shuffle_split)
        return {"train": BaseDataset(train_data, instance_transforms["train"]), 
                "test": BaseDataset(test_data, instance_transforms["test"])}
