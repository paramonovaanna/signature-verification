from torch.utils.data import Dataset

from src.datasets.loading_utils import load_pil


class BaseTransformDataset(Dataset):

    def __init__(
        self, data, instance_transforms=None
    ):
        """
        Args:
            data (list[dict], Dataset): an indexed object, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and image_path.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self.data = data
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self.data)

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
    

class NoPreprocessTD(BaseTransformDataset):

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
            label = data_dict["label"]

            img_path = data_dict["path"]
            img = load_pil(img_path)
            instance_data = {"img": img, "labels": label}

            instance_data = self.transform_data(instance_data)

            return instance_data


class PreprocessTD(BaseTransformDataset):

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
        item = self.data[ind]
        img, label = item[0], item[1]

        instance_data = {"img": img, "labels": label}
        instance_data = self.transform_data(instance_data)

        return instance_data
