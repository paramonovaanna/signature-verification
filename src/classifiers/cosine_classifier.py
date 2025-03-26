import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch

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
        super().__init__(*args, **kwargs)

    def calculate_distances(self, m):
        """
        Calculate distances between reference and sampled signatures for each person.
        
        Returns:
            Tuple of arrays containing distances and corresponding labels (1 for genuine, 0 for forged)
        """
        all_distances = []
        all_labels = []
        
        for user_id in tqdm(range(self.num_users), desc="Processing users"):
            genuine_embeddings = self.get_user_signatures(user_id, label=1)
            if len(genuine_embeddings) < m:  # Need m samples + 1 reference
                continue

            indexes = random.sample(range(len(genuine_embeddings)), m)
            samples = genuine_embeddings[indexes]
            reference = np.mean(samples, axis=0)
            
            # Calculate distances for genuine samples
            for i in range(len(genuine_embeddings)):
                if i in indexes:
                    continue
                genuine = genuine_embeddings[i]
                distance = np.linalg.norm(genuine - reference)
                all_distances.append(distance)
                all_labels.append(1)
            
            # Calculate distances for forged signatures
            forged_embeddings = self.get_user_signatures(user_id, label=0)
            
            for forged in forged_embeddings:
                distance = np.linalg.norm(forged - reference)
                all_distances.append(distance)
                all_labels.append(0)
        print(min(all_distances), max(all_distances))
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
        predictions = (distances <= self.threshold).astype(int)  # <= threshold means genuine
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