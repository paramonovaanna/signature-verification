from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from src.utils.init_utils import set_worker_seed

from src.datasets.data_utils import move_batch_transforms_to_device
from src.datasets.collate_fn import collate_fn
from src.datasets.preprocessed_dataset import PreprocessedDataset
from src.datasets.partition_dataset import PartitionDataset


class DataLoaderFactory:
    """Factory class for creating dataloaders for both regular and siamese training modes."""
    
    def __init__(self, config, device: str):
        self.config = config
        self.device = device

        self.dataset = self._create_dataset()
        self.batch_transforms = self._setup_batch_transforms()
        self.instance_transforms = self._create_instance_transforms()
        
    def _setup_batch_transforms(self):
        """Initialize and move transforms to device."""
        self.batch_transforms = instantiate(self.config.transforms.batch_transforms)
        move_batch_transforms_to_device(self.batch_transforms, self.device)
        
    def _create_dataset(self):
        """Create and preprocess the base dataset."""
        dataset = instantiate(self.config.dataset)
        preprocessor = instantiate(self.config.preprocessor)
        preprocessed_dataset = preprocessor(dataset)
        return preprocessed_dataset
        
    def _create_instance_transforms(self):
        """Create instance transforms for the model."""
        return instantiate(self.config.model.instance_transforms)
        
    def get_dataloaders(self, modes):
        """Get dataloaders for training and validation.
        
        Args:
            mode: Either "regular" or "siamese" to specify the training mode
            
        Returns:
            Tuple of (dataloaders dict, batch_transforms)
        """
        train_data, test_data = self.dataset.random_split(
            self.config.data.split, 
            self.config.data.users
        )

        datasets = {"train": PartitionDataset(train_data, self.instance_transforms.train, mode=modes[0])}
        if self.config.data.split != 1.0:
            datasets["test"] = PartitionDataset(test_data, self.instance_transforms.train, mode=modes[1])

        dataloaders = {}
        for dataset_partition in datasets.keys():
            dataset = datasets[dataset_partition]

            assert self.config.dataloaders.batch_size <= len(datasets[dataset_partition]), (
                f"The batch size ({self.config.dataloaders.batch_size}) cannot "
                f"be larger than the dataset length ({len(dataset)})"
            )

            partition_dataloader = instantiate(
                self.config.dataloaders,
                dataset=dataset,
                collate_fn=collate_fn,
                drop_last=(dataset_partition == "train"),
                shuffle=(dataset_partition == "train"),
                worker_init_fn=set_worker_seed,
            )
            dataloaders[dataset_partition] = partition_dataloader
        return dataloaders, self.batch_transforms
        
    def get_inference_dataloaders(self, mode):
        """Get dataloaders for inference.
        
        Args:
            mode: Either "standalone" or "siamese" to specify the training mode
            
        Returns:
            Tuple of (dataloaders dict, batch_transforms)
        """
        test_data, _ = self.dataset.random_split(1.0, self.config.data.users)
        
        test_dataset = PartitionDataset(test_data, self.instance_transforms.test, mode=mode)
        assert self.config.dataloaders.batch_size <= len(test_dataset), (
                f"The batch size ({self.config.dataloaders.batch_size}) cannot "
                f"be larger than the dataset length ({len(test_dataset)})"
            )
        dataloader = instantiate(self.config.dataloaders,
            dataset=test_dataset,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
            worker_init_fn=set_worker_seed
        )
   
        return {"inference": dataloader}, self.batch_transforms 