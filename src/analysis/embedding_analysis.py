import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
from tqdm import tqdm
import random
from collections import defaultdict

from src.preprocessor.htcsignet_preprocessor import HTCSigNetPreprocessor
from src.models.convnext import ConvNeXt_B
from src.datasets.basedataset import BaseDataset

class EmbeddingAnalyzer:
    def __init__(
        self,
        model_path: str,
        dataset: BaseDataset,
        preprocessor: HTCSigNetPreprocessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        num_samples_per_person: int = 5
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_samples = num_samples_per_person
        self.preprocessor = preprocessor
        
        # Load model and set to evaluation mode
        self.model = ConvNeXt_B(use_pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        # Group signatures by person
        self.person_signatures = self._group_signatures_by_person(dataset)
    
    def _group_signatures_by_person(self, dataset: BaseDataset) -> Dict[str, List[Dict]]:
        """Group signatures by person ID."""
        grouped = defaultdict(list)
        for item in dataset:
            person_id = item["person_id"]
            grouped[person_id].append(item)
        return dict(grouped)
    
    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings from the model without classification.
        
        Args:
            images: Input images tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Embeddings tensor of shape [batch_size, embedding_dim]
        """
        with torch.no_grad():
            # Get features before the classifier
            features = self.model.model.features(images)  # [batch_size, channels, h, w]
            features = self.model.model.avgpool(features)  # [batch_size, channels, 1, 1]
            features = torch.flatten(features, 1)  # [batch_size, embedding_dim]
            return features
    
    def calculate_distances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate distances between reference and sampled signatures for each person.
        
        The process for each person:
        1. Select one random reference signature
        2. Select m random genuine signatures
        3. Calculate distances between reference and all signatures
        
        Returns:
            Tuple of arrays (distances, labels) where:
            - distances: Array of euclidean distances between pairs of signatures
            - labels: Array of 1s (genuine) and 0s (forged) corresponding to distances
        """
        all_distances = []
        all_labels = []
        
        for person_id, signatures in tqdm(self.person_signatures.items(), desc="Processing people"):
            # Get genuine signatures for this person
            genuine_signatures = [s for s in signatures if s["label"] == 1]
            if len(genuine_signatures) < self.num_samples + 1:
                continue
            
            # Randomly select reference and samples
            selected = random.sample(genuine_signatures, self.num_samples + 1)
            reference = selected[0]
            samples = selected[1:]
            
            # Process reference signature
            # Shape progression for reference:
            # 1. ref_image: [height, width] (numpy array)
            # 2. After preprocess: [channels, height, width] (numpy array)
            # 3. After unsqueeze: [1, channels, height, width] (torch tensor)
            ref_image = self.preprocessor.preprocess_signature(reference["image"])
            ref_image = torch.from_numpy(ref_image).unsqueeze(0).to(self.device)
            
            # Get reference embedding
            # Shape: [1, embedding_dim]
            ref_embedding = self.get_embeddings(ref_image)
            
            # Process genuine samples
            # Shape progression for samples:
            # 1. Each sample: [channels, height, width] (numpy array)
            # 2. After stack: [num_samples, channels, height, width] (numpy array)
            # 3. After to_tensor: [num_samples, channels, height, width] (torch tensor)
            sample_images = []
            for sample in samples:
                img = self.preprocessor.preprocess_signature(sample["image"])
                sample_images.append(img)
            
            sample_images = torch.from_numpy(np.stack(sample_images)).to(self.device)
            # Shape: [num_samples, embedding_dim]
            sample_embeddings = self.get_embeddings(sample_images)
            
            # Calculate distances using torch.cdist
            # torch.cdist computes pairwise distance between two sets of vectors
            # Input shapes:
            # - ref_embedding: [1, embedding_dim]
            # - sample_embeddings: [num_samples, embedding_dim]
            # Output shape: [1, num_samples]
            distances = torch.cdist(ref_embedding, sample_embeddings).cpu().numpy()
            
            # distances[0] gives us the distances from the reference to each sample
            # Shape: [num_samples]
            genuine_distances = distances[0]
            all_distances.extend(genuine_distances)
            all_labels.extend([1] * len(genuine_distances))
            
            # Process forged signatures similarly
            forged_signatures = [s for s in signatures if s["label"] == 0]
            if forged_signatures:
                forged_images = []
                for forged in forged_signatures:
                    img = self.preprocessor.preprocess_signature(forged["image"])
                    forged_images.append(img)
                
                # Shape: [num_forged, channels, height, width]
                forged_images = torch.from_numpy(np.stack(forged_images)).to(self.device)
                # Shape: [num_forged, embedding_dim]
                forged_embeddings = self.get_embeddings(forged_images)
                
                # Shape: [1, num_forged] -> [num_forged] after [0]
                forged_distances = torch.cdist(ref_embedding, forged_embeddings).cpu().numpy()
                forged_distances = forged_distances[0]
                all_distances.extend(forged_distances)
                all_labels.extend([0] * len(forged_distances))
        
        return np.array(all_distances), np.array(all_labels)
    
    def find_optimal_threshold(self, distances: np.ndarray, labels: np.ndarray, num_steps: int = 100) -> Tuple[float, float]:
        """
        Find optimal distance threshold that maximizes accuracy.
        
        Args:
            distances: Array of distances between signatures
            labels: Array of labels (1 for genuine, 0 for forged)
            num_steps: Number of steps for threshold search
            
        Returns:
            Tuple of (optimal threshold, best accuracy)
        """
        d_min, d_max = np.min(distances), np.max(distances)
        step = (d_max - d_min) / num_steps
        thresholds = np.arange(d_min, d_max + step, step)
        
        best_accuracy = 0
        best_threshold = 0
        
        for threshold in thresholds:
            predictions = (distances < threshold).astype(int)
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold, best_accuracy

def main():
    # Example usage
    from src.datasets.cedar import CEDARDataset
    from src.configs.baseline import get_config
    
    config = get_config()
    
    # Initialize dataset and preprocessor
    dataset = CEDARDataset(**config.dataset)
    preprocessor = HTCSigNetPreprocessor(
        canvas_size=config.dataset.canvas_size,
        img_size=config.dataset.img_size,
        input_size=config.dataset.input_size,
        dataset=dataset
    )
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(
        model_path="path/to/your/model.pth",
        dataset=dataset,
        preprocessor=preprocessor,
        num_samples_per_person=5
    )
    
    # Calculate distances
    distances, labels = analyzer.calculate_distances()
    
    # Find optimal threshold
    optimal_threshold, best_accuracy = analyzer.find_optimal_threshold(distances, labels)
    
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main() 