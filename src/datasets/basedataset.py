import random

import safetensors
import safetensors.torch
import torch
from torch.utils.data import Dataset


from tqdm import tqdm


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self, data, instance_transforms=None
    ):
        """
        Args:
            data (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and loaded image.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_data_is_valid(data)
        self.data = data

        self.instance_transforms = instance_transforms

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
        data_dict = self.data[ind]

        img = data_dict["img"]
        label = data_dict["label"]

        instance_data = {"img": img, "labels": label}
        instance_data = self.transform_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def transform_data(self, instance_data):
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
    
    def preprocess_data(self, preprocessor):
        print("Preprocessing images...")
        for i in tqdm(range(len(self.data))):
            img = self.data[i]["img"]
            self.data[i]["img"] = preprocessor(img)

    @staticmethod
    def _assert_data_is_valid(data):
        """
        Check the structure of the data and ensure it satisfies the desired
        conditions.

        Args:
            data (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and loaded image.
        """
        for entry in data:
            assert "img" in entry, (
                "Each dataset item should include field 'img'" " - a loaded image file."
            )
            assert "label" in entry, (
                "Each dataset item should include field 'label'"
                " - object ground-truth label."
            )
