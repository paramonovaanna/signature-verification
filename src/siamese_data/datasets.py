import torch
from torch.utils.data import Dataset

import numpy as np
import random

class SiameseTrainDataset(Dataset):
    def __init__(self, data, instance_transforms):
        """
        Dataset for signature verification using Siamese Networks.
        Args:
            data: List[List[Dict]]
            phase: 'train' or 'val'
            instance_transforms: Instance transforms
        """
        super().__init__()
        self.base_data = data
        self.num_users = len(self.base_data)
        self.instance_transforms = instance_transforms
        
        self.anchors = []
        self.positive_imgs = []
        self.negative_imgs = []
        self.idx2users = {}

        self._create_triplets()

    def _create_triplets(self):
        """
        Each genuine signature becomes an anchor only once. As a positive img a genuine
        signature is randomly selected from the users genuine signatures. The same is done
        with choosing a negative img.
        """
        previous_data = 0

        for user_id in range(self.num_users):
            genuine = [sig for sig in self.base_data[user_id] if sig['labels'] == 1]
            forged = [sig for sig in self.base_data[user_id] if sig['labels'] == 0]
            
            for idx, anchor in enumerate(genuine):
                positive = random.choice([genuine[i] for i in range(len(genuine)) if i != idx])
                negative = random.choice(forged)
                
                self.anchors.append(anchor)
                self.positive_imgs.append(positive)
                self.negative_imgs.append(negative)
                self.idx2users[previous_data + idx] = user_id
                
            previous_data += len(genuine)

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
    
    def __getitem__(self, idx):
        anchor_img = self.anchors[idx]
        positive_img = self.positive_imgs[idx]
        negative_img = self.negative_imgs[idx]
        
        if self.instance_transforms:
            anchor_img = self.transform_data(anchor_img)
            positive_img = self.transform_data(positive_img)
            negative_img = self.transform_data(negative_img)
            
        return {
            'anchor': anchor_img["img"],
            'positive': positive_img["img"],
            'negative': negative_img["img"],
        } 
    
    def __len__(self) -> int:
        return len(self.anchors)


class SiameseTestDataset(Dataset):
    def __init__(self, data, instance_transforms):
        super().__init__()
        self.base_data = data
        self.instance_transforms = instance_transforms

        self.num_users = len(self.base_data)

        self.pairs = []
        self._create_pairs()

    def _create_pairs(self):
        for user_id in range(self.num_users):
            genuine = [sig for sig in self.base_data[user_id] if sig['labels'] == 1]
            reference_idx = random.choice(np.arange(len(genuine)))
            forged = [sig for sig in self.base_data[user_id] if sig['labels'] == 0]

            reference = genuine[reference_idx]

            for i in range(len(genuine)):
                if i == reference_idx:
                    continue
                self.pairs.append((reference, genuine[i]))

            for forg_sig in forged:
                self.pairs.append((reference, forg_sig))

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

    def __getitem__(self, idx):
        reference, sig = self.pairs[idx]
        if self.instance_transforms:
            reference = self.transform_data(reference)
            sig = self.transform_data(sig)
        return {
            "reference": reference["img"],
            "sig": sig["img"], 
            "labels": sig["labels"]
        }

    def __len__(self) -> int:
        return len(self.pairs)
            
