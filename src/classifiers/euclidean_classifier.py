import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch

import random

from srs.classifiers.distance_classifier import DistanceClassifier


class EucledianClassifier(DistanceClassifier):
    def __init__(self, num_steps, *args, **kwargs):
        """
        Initialize DistanceClassifier.
        
        Args:
            dataset: PreprocessedDataset instance containing signature data
        """
        self.num_steps = num_steps
        self.reference_idx = []
        super().__init__(*args, **kwargs)

    def calculate_distances(self, m):
        """
        Calculate distances between reference and sampled signatures for each person.
        
        Returns:
            Tuple of arrays containing distances and corresponding labels (1 for genuine, 0 for forged)
        """
        all_distances = []
        all_labels = []
        D = []
        
        for user_id in tqdm(range(self.num_users), desc="Processing users"):
            genuine_embeddings = self.get_user_signatures(user_id, label=1)
            if len(genuine_embeddings) < m + 1:  # Need m samples + 1 reference
                continue
                
            valid_indexes = [i for i in range(len(genuine_embeddings)) if i != self.reference_idx[user_id]]
            samples = random.sample(valid_indexes, m)
            reference = genuine_embeddings[self.reference_idx[user_id]]
            
            # Calculate distances for genuine samples
            for i in range(len(genuine_embeddings)):
                if i == self.reference_idx[user_id]:
                    continue
                genuine = genuine_embeddings[i]
                distance = np.linalg.norm(genuine - reference)
                all_distances.append(distance)
                all_labels.append(1)
                if i in samples:
                    D.append(distance)
            
            # Calculate distances for forged signatures
            forged_embeddings = self.get_user_signatures(user_id, label=0)
            
            for forged in forged_embeddings:
                distance = np.linalg.norm(forged - reference)
                all_distances.append(distance)
                all_labels.append(0)
        
        return np.array(all_distances), np.array(all_labels), np.array(D)
    
    def find_optimal_threshold(self, distances: np.ndarray, labels: np.ndarray, D) -> Tuple[float, float]:
        """
        Find optimal distance threshold that maximizes accuracy.
        
        Args:
            distances: Array of distances between signatures
            labels: Array of labels (1 for genuine, 0 for forged)
            num_steps: Number of steps for threshold search
            
        Returns:
            Tuple of (optimal threshold, best accuracy)
        """
        d_min, d_max = np.min(D), np.max(D)
        step = (d_max - d_min) / self.num_steps
        thresholds = np.arange(d_min, d_max + step, step)
        
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in tqdm(thresholds, desc="Finding optimal threshold"):
            predictions = (distances <= threshold).astype(int)  # <= threshold means genuine
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy

    def fix_references(self):
        for user_id in range(self.num_users):
            genuine_sigs = self.get_user_signatures(user_id, label=1)
            index = random.choice(range(len(genuine_sigs)))
            self.reference_idx.append(index)
    
    def classify(self) -> Dict[str, Any]:
        """
        Analyze signatures of all users and find optimal threshold.
        
        Returns:
            Dictionary containing analysis results
        """
        logs = []
        for m in self.m_values:
            # Calculate distances
            distances, labels, D = self.calculate_distances(m)
            
            # Find optimal threshold
            optimal_threshold, best_accuracy = self.find_optimal_threshold(distances, labels, D)
            
            logs.append({
                'm': m,
                'optimal_threshold': optimal_threshold,
                'best_accuracy': best_accuracy,
            })
        return logs