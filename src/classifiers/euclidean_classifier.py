import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch

import random

from src.classifiers.distance_classifier import DistanceClassifier


class EuclideanClassifier(DistanceClassifier):
    def __init__(self, num_steps, user2emb, references=None, **kwargs):
        """
        Initialize DistanceClassifier.
        
        Args:
            dataset: PreprocessedDataset instance containing signature data
        """
        super().__init__(user2emb, **kwargs)
        self.num_steps = num_steps
        self._fix_references(user2emb, references)

    def _init_user_embeddings(self, embeddings, user2emb, labels):
        user_embeddings = [{"genuine": [], "forged": []} for i in range(self.num_users)]

        for i in range(len(user2emb)):
            user_id = user2emb[i]
            label = labels[i]
            emb = embeddings[i]
            if label == 1:
                user_embeddings[user_id]["genuine"].append(emb)
            else:
                user_embeddings[user_id]["forged"].append(emb)
        return user_embeddings
        
    def _fix_references(self, user2emb, references):
        if references is not None:
            for i in range(len(user2emb)):
                user_id = user2emb[i]
                self.user_embeddings[user_id]["reference"] = references[i]
        else:
            for i in range(len(self.num_users)):
                genuine = self.user_embeddings[i]["genuine"]
                ref_idx = random.choice(range(len(genuine)))
                self.user_embeddings[user_id]["reference"] = self.user_embeddings[i]["genuine"].pop(ref_idx)

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
            genuine_embeddings = self.user_embeddings[user_id]["genuine"]
            if len(genuine_embeddings) < m:
                continue
            samples = random.sample(range(len(genuine_embeddings)), m)
            reference = self.user_embeddings[user_id]["reference"]
            
            # Calculate distances for genuine samples
            for i in range(len(genuine_embeddings)):
                genuine = genuine_embeddings[i]
                distance = np.linalg.norm(genuine - reference)
                all_distances.append(distance)
                all_labels.append(1)
                if i in samples:
                    D.append(distance)
            
            # Calculate distances for forged signatures
            forged_embeddings = self.user_embeddings[user_id]["forged"]
            
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