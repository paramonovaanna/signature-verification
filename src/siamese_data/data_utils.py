import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Union
import os
import glob
from random import sample
from PIL import Image
import torchvision.transforms as transforms

import hydra
from hydra.utils import instantiate

from src.siamese_data.collate_fn import train_collate_fn, test_collate_fn
from src.siamese_data.datasets import SiameseTrainDataset, SiameseTestDataset

from src.utils.init_utils import set_worker_seed

from src.datasets import move_batch_transforms_to_device


def get_inference_dataloaders(config, device):
    """
    Create dataloaders for inference.
    
    Args:
        data_root: Path to the dataset root directory
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    dataset = instantiate(config.dataset)

    preprocessor = instantiate(config.preprocessor)
    preprocessed_dataset = preprocessor(dataset)

    instance_transforms = instantiate(config.model.instance_transforms)
    test_data, _ = preprocessed_dataset.random_split(1.0, config.train_test.users)

    dataset = SiameseTestDataset(test_data, instance_transforms.test)

    assert config.dataloaders.batch_size <= len(dataset), (
        f"The batch size ({config.dataloaders.batch_size}) cannot "
        f"be larger than the dataset length ({len(dataset)})"
    )

    partition_dataloader = instantiate(
        config.dataloaders,
        dataset=dataset,
        collate_fn=test_collate_fn,
        drop_last=False,
        shuffle=False,
        worker_init_fn=set_worker_seed,
    )
    return {"inference": partition_dataloader}, batch_transforms


def get_dataloaders(config, device):
    """
    Create dataloaders for training and validation.
    
    Args:
        data_root: Path to the dataset root directory
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    dataset = instantiate(config.dataset)

    preprocessor = instantiate(config.preprocessor)
    preprocessed_dataset = preprocessor(dataset)

    instance_transforms = instantiate(config.model.instance_transforms)
    train_data, test_data = preprocessed_dataset.random_split(config.train_test.split, config.train_test.users)

    datasets = {
        "train": SiameseTrainDataset(train_data, instance_transforms.train),
        "test": SiameseTestDataset(test_data, instance_transforms.test)
    } 
    collate_fn = {"train": train_collate_fn, "test": test_collate_fn}

    dataloaders = {}
    for dataset_partition in datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloaders.batch_size <= len(dataset), (
            f"The batch size ({config.dataloaders.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloaders,
            dataset=dataset,
            collate_fn=collate_fn[dataset_partition],
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms