import random
from src.datasets.basedataset import BaseDataset

class BaseDownloader:

    def __init__(self, index, limit=None, shuffle_limit=False):
        self._index = index
        if limit is not None:
            self._index = self._limit_dataset(limit, shuffle_limit)

    def _random_split(self, split, shuffle_split):
        assert 0 <= split <= 1, ("Split should be between 0 and 1")

        train_size = int(len(self._index) * split)

        if shuffle_split:
            random.seed(42)
            random.shuffle(self._index)
        return self._index[:train_size], self._index[train_size:]
    
    def _limit_dataset(self, limit, shuffle_limit):
        if shuffle_limit:
            random.seed(42)
            random.shuffle(self._index)

        index = self._index[:limit]
        return index
    
    def get_partitions(self, split, shuffle_split, instance_transforms=None):
        '''
        Returns a partitions dict object
        '''
        train_index, test_index = self._random_split(split, shuffle_split)
        return {"train": BaseDataset(train_index, instance_transforms["train"]), 
                "test": BaseDataset(test_index, instance_transforms["test"])}
