import torch
from torch.utils.data import Dataset

import numpy as np
import random

class PartitionDataset(Dataset):
    def __init__(self, base_data, instance_transforms, mode="standalone"):
        """
        Dataset for signature verification using Siamese Networks.
        Args:
            data: List[List[Dict]]
            phase: 'train' or 'val'
            instance_transforms: Instance transforms
        """
        super().__init__()
        self.num_users = len(base_data)
        self.instance_transforms = instance_transforms

        self.data = []
        if mode == "triplets":
            self._create_triplets(base_data)
        elif mode == "pairs":
            self._create_pairs(base_data)
        elif mode == "singles":
            self.data = self._create_singles(base_data)

    def _create_singles(self, base_data):
        data = []
        for user_id in range(self.num_users):
            for i in range(len(base_data[user_id])):
                instance = base_data[user_id][i]
                instance["user"] = user_id
                data.append(instance)
        return data

    def _create_triplets(self, base_data):
        """
        Each genuine signature becomes an anchor only once. As a positive img a genuine
        signature is randomly selected from the users genuine signatures. The same is done
        with choosing a negative img.
        """
        for user_id in range(self.num_users):
            genuine = [sig for sig in base_data[user_id] if sig['labels'] == 1]
            forged = [sig for sig in base_data[user_id] if sig['labels'] == 0]

            triplet = {"user": user_id}
            for idx, anchor in enumerate(genuine):
                positive = random.choice([genuine[i] for i in range(len(genuine)) if i != idx])
                negative = random.choice(forged)

                triplet["anchor"] = anchor["img"]
                triplet["pos"] = positive["img"]
                triplet["neg"] = negative["img"]
                self.data.append(triplet)

    def _create_pairs(self, base_data):
        for user_id in range(self.num_users):
            genuine = [sig for sig in base_data[user_id] if sig['labels'] == 1]
            reference_idx = random.choice(np.arange(len(genuine)))
            forged = [sig for sig in base_data[user_id] if sig['labels'] == 0]

            reference = genuine[reference_idx]
            for i in range(len(genuine)):
                if i == reference_idx:
                    continue
                pair = {"user": user_id, "ref": reference["img"], "sig": genuine[i]["img"], "labels": 1}
                self.data.append(pair)

            for forg_sig in forged:
                pair = {"user": user_id, "ref": reference["img"], "sig": forg_sig["img"], "labels": 0}
                self.data.append(pair)

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
            instance_data = self.instance_transforms(instance_data)
        return instance_data
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        for key in instance.keys():
            if key == "user" or key == "labels":
                continue
            instance[key] = self.transform_data(instance[key])
        return instance
    
    def __len__(self) -> int:
        return len(self.data)
