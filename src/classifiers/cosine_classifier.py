import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn

import random

from src.classifiers.distance_classifier import DistanceClassifier


class CosineClassifier(DistanceClassifier):

    def __init__(self, threshold, *args, **kwargs):
        """
        Initialize DistanceClassifier.
        
        Args:
            dataset: PreprocessedDataset instance containing signature data
        """
        self.threshold = threshold
        self.similarity_fn = nn.CosineSimilarity(dim=1, eps=1e-8)

        super().__init__(*args, **kwargs)

    def _init_user_embeddings(self, embeddings, user2emb, labels, ref_embeddings=None):
        user_embeddings = [{"genuine": [], "forged": []} for i in range(self.num_users)]

        for i in range(len(user2emb)):
            user_id = user2emb[i]
            label = labels[i]
            emb = embeddings[i]
            if label == 1:
                user_embeddings[user_id]["genuine"].append(emb)
            else:
                user_embeddings[user_id]["forged"].append(emb)
        if ref_embeddings is not None:
            for i in range(self.num_users):
                idx = user2emb.index(i)
                user_embeddings[i]["geniune"].append(ref_embeddings[idx])
        return user_embeddings

    def calculate_distances(self, m):
        """
        Calculate distances between reference and sampled signatures for each person.
        
        Returns:
            Tuple of arrays containing distances and corresponding labels (1 for genuine, 0 for forged)
        """
        all_distances = []
        all_labels = []
        
        for user_id in tqdm(range(self.num_users), desc="Processing users"):
            genuine_embeddings = self.user_embeddings[user_id]["genuine"]
            if len(genuine_embeddings) < m:  # Need m samples
                continue

            indexes = random.sample(range(len(genuine_embeddings)), m)
            samples = [genuine_embeddings[i] for i in indexes]
            reference = torch.from_numpy(np.mean(samples, axis=0)).unsqueeze(0)
            
            # Calculate distances for genuine samples
            for i in range(len(genuine_embeddings)):
                if i in indexes:
                    continue
                genuine = genuine_embeddings[i]
                distance = self.similarity_fn(torch.from_numpy(genuine).unsqueeze(0), reference)
                all_distances.append(distance)
                all_labels.append(1)
            
            # Calculate distances for forged signatures
            forged_embeddings = self.user_embeddings[user_id]["forged"]
            
            for forged in forged_embeddings:
                distance = self.similarity_fn(torch.from_numpy(forged).unsqueeze(0), reference)
                all_distances.append(distance)
                all_labels.append(0)
        return np.array(all_distances), np.array(all_labels)
    
    def calculate_accuracy(self, distances: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Find optimal distance threshold that maximizes accuracy.
        
        Args:
            distances: Array of distances between signatures
            labels: Array of labels (1 for genuine, 0 for forged)
            num_steps: Number of steps for threshold search
            
        Returns:
            Tuple of (optimal threshold, best accuracy)
        """
        probs = (distances + 1) / 2
        predictions = (distances >= self.threshold).astype(int)  # <= threshold means genuine
        accuracy = np.mean(predictions == labels)        
        return accuracy
    
    def classify(self) -> Dict[str, Any]:
        """
        Analyze signatures of all users and find optimal threshold.
        
        Returns:
            Dictionary containing analysis results
        """
        logs = []
        for m in self.m_values:
            # Calculate distances
            distances, labels = self.calculate_distances(m)
            
            # Find optimal threshold
            accuracy = self.calculate_accuracy(distances, labels)
            
            logs.append({
                'm': m,
                'threshold': self.threshold,
                'accuracy': accuracy,
            })
        return logs