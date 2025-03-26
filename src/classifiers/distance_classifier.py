import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch

import random


class DistanceClassifier:
    def __init__(self, 
                device,
                device_tensors,
                model, 
                num_users,
                dataloader,
                batch_transforms, *args, **kwargs):
        """
        Initialize DistanceClassifier.
        
        Args:
            dataset: PreprocessedDataset instance containing signature data
        """
        self.dataloader = dataloader
        self.batch_transforms = batch_transforms

        self.num_users = num_users

        self.device = device
        self.device_tensors = device_tensors
        
        # Load model
        self.model = model
        self._from_pretrained(config.from_pretrained)
        self.model.eval()
        
        # Initialize storage for embeddings
        self.embeddings = [[] for _ in range(self.num_users)]
        self.labels = [[] for _ in range(self.num_users)]

        self.num_steps = num_steps

    def extract_embeddings(self):
        print("Extracting embeddings...")

        for batch_idx, batch in enumerate(
            tqdm(self.dataloader, total=len(self.dataloader))
        ):
            with torch.no_grad():
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)

                embeddings = self.model.features(**batch)
                embeddings["emb"] = embeddings["emb"].mean(dim=[2, 3]) # 32 * 1024
                batch.update(embeddings)

                for i in range(len(batch["labels"])):
                    user_idx = batch["user"][i]
                    emb = batch["emb"][i].cpu().numpy()
                    label = np.array([batch["labels"][i].item()])
                    self.embeddings[user_idx].append(emb)
                    self.labels[user_idx].append(label)

        for user_idx in range(self.num_users):
            self.embeddings[user_idx] = np.array(self.embeddings[user_idx])
            self.labels[user_idx] = np.concatenate(self.labels[user_idx])

    def get_user_signatures(self, idx, label=1):
        user_embeddings = self.embeddings[idx]
        user_labels = self.labels[idx]
        genuine_mask = (user_labels == label)
        genuine_embeddings = user_embeddings[genuine_mask]
        return genuine_embeddings
    
    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        # do batch transforms on device
        transform_type = "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch
    
    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)
        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)