import random

import safetensors
import safetensors.torch
import torch
from torch.utils.data import Dataset

from PIL import Image

import numpy as np

from skimage.io import imread
from skimage import img_as_ubyte


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self, index, instance_transforms=None, load_numpy=False
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)
        self._index = index

        self.instance_transforms = instance_transforms
        self.load_numpy = load_numpy

    def __getitem__(self, ind):
        """
        Get element from the index[partition], preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]

        if self.load_numpy:
            img = self.load_img_numpy(data_path)
        else:
            img = self.load_img(data_path)
        label = data_dict["label"]

        instance_data = {"img": img, "labels": label}
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

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
    
    def load_img(self, path):
        """
        Load img from disk.

        Args:
            path (str): path to the object.
        Returns:
            img (Tensor):
        """
        img = Image.open(path).convert("L")
        return img

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to image file."
            )
            assert "label" in entry, (
                "Each dataset item should include field 'label'"
                " - object ground-truth label."
            )
