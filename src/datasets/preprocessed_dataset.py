import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import random

import torch

class PreprocessedDataset:
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indexes: np.ndarray,
        user_mapping: Dict[int, str]
    ):
        """
        Initialize PreprocessTD to convert array format to list of dictionaries.
        
        Args:
            images: np.ndarray (N x 1 x H x W) grayscale signature images
            labels: np.ndarray (N) forgery labels (0: skilled forgery, 1: genuine)
            indexes: np.ndarray (N) user indices
            user_mapping: dict mapping indices to original user IDs
        """
        self.images = images
        self.labels = labels
        self.indexes = indexes
        self.user_mapping = user_mapping
        
        # Create reverse mapping from original user IDs to indices
        self.reverse_mapping = {v: k for k, v in user_mapping.items()}
        
        # Convert data to required format
        self.data = self._process_data()
        
    def _process_data(self):
        """
        Convert array data to list of lists of dictionaries format.
        Each sublist contains dictionaries with signatures for one user.
        
        Returns:
            List[List[Dict]]: List of lists where each sublist contains
            dictionaries with 'image' and 'label' keys for a specific user
        """
        num_users = len(self.user_mapping)
        result = [[] for _ in range(num_users)]
        
        for i in range(len(self.images)):
            user_idx = self.indexes[i]
            signature_dict = {
                "img": self.images[i],
                "labels": int(self.labels[i])
            }
            result[user_idx].append(signature_dict)
            
        return result
    
    def random_split(self, split: float, user_indices: List[int]):
        """
        Randomly split users into two groups and collect their signatures.
        
        Args:
            split: float between 0 and 1, proportion of users for first split
            user_indices: list of user indices to include in the split
            
        Returns:
            Tuple of two lists of dictionaries, each dictionary contains 'img' and 'label' keys.
            First list contains signatures from 'split' proportion of users, 
            second list contains signatures from the rest of users.
        """
        assert 0 < split < 1, ("Split must be between 0 and 1")
        
        # Shuffle user indices
        random.shuffle(user_indices)
        
        # Split users into two groups
        split_idx = int(len(user_indices) * split)
        first_group = user_indices[:split_idx]
        second_group = user_indices[split_idx:]
        
        first_split = []
        for user_idx in first_group:
            first_split.extend([
                {
                    "img": sig["img"],
                    "label": sig["labels"]
                }
                for sig in self.data[user_idx]
            ])
            
        second_split = []
        for user_idx in second_group:
            second_split.extend([
                {
                    "img": sig["img"],
                    "label": sig["labels"]
                }
                for sig in self.data[user_idx]
            ])
        
        return first_split, second_split
    
    def get_user_signatures(self, user_id: str) -> Optional[List[Dict]]:
        """
        Get all signatures for a specific user by their original ID.
        
        Args:
            user_id: Original user ID from the dataset
            
        Returns:
            List of dictionaries containing signatures and labels for the user,
            or None if user not found
        """
        if user_id not in self.reverse_mapping:
            return None
        user_idx = self.reverse_mapping[user_id]
        return self.processed_data[user_idx]
    
    def get_genuine_signatures(self, user_id: str) -> Optional[List[Dict]]:
        """
        Get only genuine signatures for a specific user.
        
        Args:
            user_id: Original user ID from the dataset
            
        Returns:
            List of dictionaries containing only genuine signatures,
            or None if user not found
        """
        signatures = self.get_user_signatures(user_id)
        if signatures is None:
            return None
        return [s for s in signatures if s["label"] == 1]
    
    def get_forged_signatures(self, user_id: str) -> Optional[List[Dict]]:
        """
        Get only forged signatures for a specific user.
        
        Args:
            user_id: Original user ID from the dataset
            
        Returns:
            List of dictionaries containing only forged signatures,
            or None if user not found
        """
        signatures = self.get_user_signatures(user_id)
        if signatures is None:
            return None
        return [s for s in signatures if s["label"] == 0]
    
    def __len__(self) -> int:
        """Return number of users in the dataset."""
        return len(self.user_mapping)
    
    def __getitem__(self, idx: int) -> List[Dict]:
        """Get signatures for user by index."""
        return self.processed_data[idx] 