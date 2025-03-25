from hydra.utils import instantiate

import torch

from itertools import repeat

from torch.utils.data import TensorDataset, random_split

from src.datasets import TransformDataset

from src.datasets.collate_fn import collate_fn
from src.utils.init_utils import set_worker_seed

def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if not batch_transforms:
        return
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_inference_dataloaders(config, device):
    """Create dataloaders for testing.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for 
            testing.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """

    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    dataset = instantiate(config.dataset)

    preprocessor = instantiate(config.preprocessor)
    preprocessed_dataset = preprocessor(dataset)

    instance_transforms = instantiate(config.model.instance_transforms)
    test_data, _ = preprocessed_dataset.random_split(1.0, config.test.users)

    test_dataset = TransformDataset(test_data, instance_transforms)

    # dataloaders init
    dataloaders = {}

    assert config.dataloaders.batch_size <= len(test_dataset), (
        f"The batch size ({config.dataloaders.batch_size}) cannot "
        f"be larger than the dataset length ({len(dataset)})"
    )

    partition_dataloader = instantiate(
        config.dataloaders,
        dataset=test_dataset,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        worker_init_fn=set_worker_seed,
    )

    dataloaders["inference"] = partition_dataloader

    return dataloaders, batch_transforms


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    dataset = instantiate(config.dataset)

    preprocessor = instantiate(config.preprocessor)
    preprocessed_dataset = preprocessor(dataset)

    instance_transforms = instantiate(config.model.instance_transforms)
    train_data, test_data = preprocessed_dataset.random_split(config.train_test.split, config.train_test.users)

    datasets = {
        "train": TransformDataset(train_data, instance_transforms),
        "test": TransformDataset(test_data, instance_transforms)
    } 

    # dataloaders init
    dataloaders = {}
    for dataset_partition in datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloaders.batch_size <= len(datasets[dataset_partition]), (
            f"The batch size ({config.dataloaders.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloaders,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
