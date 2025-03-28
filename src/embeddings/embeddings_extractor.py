import torch
import os
import numpy as np

from tqdm import tqdm

from src.utils.io_utils import ROOT_PATH

class EmbeddingsExtractor:

    def __init__(self, models, pretrained_paths, device, device_tensors=["img, labels"]):   
        self.model1 = self._init_model(models[0], pretrained_paths[0])
        self.model2 = None
        if models[1] is not None:
            self.model2 = self._init_model(models[1], pretrained_paths[1])

        self.device = device
        self.device_tensors = device_tensors

    def extract(self, save_dir, filename, dataloaders=None):
        filepath = self._get_path(save_dir, filename)

        if not os.path.exists(filepath):
            assert self.model1 is not None, ("Must provide a model")
            assert dataloaders is not None, ("Must provide dataloaders")
            assert self.device is not None, ("Must provide a device")

            print("Extracting embeddings...")
            emb, labels = self.extract_and_save(filepath, dataloaders)
        else:
            emb, labels = self.load(filepath)
        return emb, labels

    def load(self, filepath):
        emb, labels = {}, {}
        with np.load(filepath, allow_pickle=True) as data:
            for partition in ["train", "val", "test"]:
                emb[partition] = data[f"{partition}_embeddings"]
                labels[partition] = data[f"{partition}_labels"]
        return emb, labels

    def extract_and_save(self, filepath, dataloaders):
        emb, labels = {}, {}
        for partition in ["train", "val", "test"]:
            emb[partition], labels[partition] = self.extract_embeddings(dataloaders[partition])

        np.savez(
            filepath,
            train_embeddings=emb["train"],
            val_embeddings=emb["val"],
            test_embeddings=emb["test"],
            train_labels=labels["train"],
            val_labels=labels["val"],
            test_labels=labels["test"]
        )
        return emb, labels

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
        self.model1.eval()
        if self.model2 is not None:
            self.model2.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting embeddings..."):
                batch = self.move_batch_to_device(batch)

                outputs = self.model1.features(**batch)["emb"]
                if self.model2 is not None:
                    outputs2 = self.model2.features(**batch)["emb"]
                    outputs = 0.5 * (outputs + outputs2)

                embeddings.append(outputs.cpu().numpy())

                labels.append(batch['labels'].cpu().numpy())

            embeddings = np.vstack(embeddings)
            labels = np.concatenate(labels)
       
        return embeddings, labels

    def _init_model(self, model, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        if pretrained_path is None:
            return model
        pretrained_path = str(pretrained_path)
        print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)
        if checkpoint.get("state_dict") is not None:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model
    
    def _get_path(self, save_dir, filename):
        path = ROOT_PATH / "data" / "embeddings" / save_dir
        os.makedirs(path, exist_ok=True)
        if not self.filename:
            filename = f"{self.models[0].name}_{self.models[1].name if self.models[1] is not None else ""}.npz"
        filepath = os.path.join(save_dir, filename)
        return filepath
    
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
        for tensor_for_device in self.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch