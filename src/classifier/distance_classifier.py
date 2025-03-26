import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
import torch

import random


class DistanceClassifier:
    def __init__(self, 
                config,
                device,
                model, 
                num_users,
                dataloader,
                batch_transforms,
                num_steps=20):
        """
        Initialize DistanceClassifier.
        
        Args:
            dataset: PreprocessedDataset instance containing signature data
        """
        self.dataloader = dataloader
        self.batch_transforms = batch_transforms

        self.num_users = num_users

        self.device = device
        
        # Load model
        self.model = model
        self._from_pretrained(config.classifier.pretrained_path)
        self.model.eval()
        
        # Initialize storage for embeddings
        self.embeddings = [[] for _ in range(self.num_users)]
        self.labels = [[] for _ in range(self.num_users)]

        self.num_steps = num_steps

    def extract_embeddings(self):
        print("Extracting embeddings...")

        for batch_idx, batch in tqdm(enumerate(self.dataloader)):
            with torch.no_grad():
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)

                embeddings = self.model.features(**batch)
                embeddings["emb"] = embeddings["emb"].mean(dim=[2, 3]) 
                batch.update(embeddings)

                for i in range(len(batch["labels"])):
                    user_idx = batch["user"][i]
                    emb = batch["emb"][i].cpu().numpy()
                    label = batch["labels"][i].item()
                    self.embeddings[user_idx].append(emb)
                    self.labels[user_idx].append(label)

        for user_idx in len(self.num_users):
            self.embeddings[user_idx] = np.concatenate(self.embeddings[user_idx])
    
    def calculate_distances(self, m):
        """
        Calculate distances between reference and sampled signatures for each person.
        
        Returns:
            Tuple of arrays containing distances and corresponding labels (1 for genuine, 0 for forged)
        """
        all_distances = []
        all_labels = []
        D = []
        
        for user_id in tqdm(len(self.embeddings_data), desc="Processing users"):
            user_embeddings = self.embeddings[user_id]["emb"]
            user_labels = self.labels[user_id]
            
            genuine_mask = (user_labels == 1)
            genuine_embeddings = user_embeddings[genuine_mask]
            
            if len(genuine_embeddings) < m + 1:  # Need m samples + 1 reference
                continue
                
            # Randomly select reference and samples
            selected_indices = random.sample(range(len(genuine_embeddings)), m + 1)
            reference = genuine_embeddings[selected_indices[0]]
            samples = selected_indices[1:]
            
            # Calculate distances for genuine samples
            for i in range(len(genuine_embeddings)):
                genuine = genuine_embeddings[i]
                distance = np.linalg.norm(genuine - reference)
                all_distances.append(distance)
                all_labels.append(1)
                if i in samples:
                    D.append(distance)
            
            # Calculate distances for forged signatures
            forged_mask = not genuine_mask
            forged_embeddings = user_embeddings[forged_mask]
            
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
    
    def classify(self, m) -> Dict[str, Any]:
        """
        Analyze signatures of all users and find optimal threshold.
        
        Returns:
            Dictionary containing analysis results
        """
        # Calculate distances
        distances, labels, D = self.calculate_distances(m)
        
        # Find optimal threshold
        optimal_threshold, best_accuracy = self.find_optimal_threshold(distances, labels, D)
        
        return {
            'optimal_threshold': optimal_threshold,
            'best_accuracy': best_accuracy,
        }
    
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
        for tensor_for_device in self.cfg_trainer.device_tensors:
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
        transform_type = "train" if self.is_train else "inference"
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