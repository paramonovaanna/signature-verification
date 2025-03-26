import torch
import os
import numpy as np

from tqdm import tqdm

from utils.io_utils import ROOT_PATH

class EmbeddingsExtractor:

    def __init__(self, save_dir, filename, model=None, dataloaders=None, device=None):
        path = ROOT_PATH / "embeddings" / save_dir
        os.makedirs(path, exist_ok=True)
        if not self.filename:
            filename = f"{model.name}.npz"
        self.filepath = os.path.join(save_dir, filename)
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device

    def extract(self):
        if not os.path.exists(self.filepath):
            assert self.model is not None, ("Must provide a model")
            assert self.dataloaders is not None, ("Must provide dataloaders")
            assert self.device is not None, ("Must provide a device")

            print("Extracting embeddings...")
            train_emb, train_labels, test_emb, test_labels = self.extract_and_save_embeddings()
        else:
            train_emb, train_labels, test_emb, test_labels = self.load_embeddings()
        return train_emb, train_labels, test_emb, test_labels
    
    def load_embeddings(self):
        with np.load(self.filepath, allow_pickle=True) as data:
            train_emb, train_labels = data["train_embeddings"], data["train_labels"]
            test_emb, test_labels = data["test_embeddings"], data["test_labels"]

        return train_emb, train_labels, test_emb, test_labels

    def extract_and_save(self):
        train_emb, train_labels = self.extract_embeddings(self.dataloaders["train"])
        test_emb, test_labels = self.extract_embeddings(self.dataloaders["test"])

        np.savez(
            self.filepath,
            train_embeddings=train_emb,
            train_labels=train_labels,
            test_embeddings=test_emb,
            test_labels=test_labels
        )
        return train_emb, train_labels, test_emb, test_labels

    def extract_embeddings(self, dataloader):
        """
        Извлекает эмбеддинги из модели и сохраняет их
        
        Args:
            model: модель PyTorch
            dataloader: DataLoader с данными
            model_name: имя модели
            phase: фаза ('train' или 'val')
            device: устройство для вычислений
            save_dir: директория для сохранения
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: эмбеддинги и метки
        """
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting embeddings..."):
                batch = self.move_batch_to_device(batch)

                outputs = self.model.features(**batch)["emb"]
                embeddings.append(outputs.cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())

            embeddings = np.vstack(embeddings)
            labels = np.concatenate(labels)
       
        return embeddings, labels
    
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