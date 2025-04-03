import torch
import os
import numpy as np

from tqdm import tqdm

from src.utils.io_utils import ROOT_PATH

class EmbeddingsExtractor:

    DATA_STRUCTURE = {
        "triplets": ["a_emb", "n_emb", "p_emb", "user"],
        "pairs": ["r_emb", "s_emb", "labels", "user"],
        "singles": ["emb", "labels", "user"]
    }

    def __init__(self, config,
        model, 
        data_mode,
        device, 
        dataloaders=None, 
    ):  

        self.device = device
        self.device_tensors = config.device_tensors

        self.model = self._init_model(model, config.from_pretrained)

        self.dataloaders = dataloaders
        self.data_mode = data_mode
        self.data_struct = self.DATA_STRUCTURE[data_mode]

    def extract(self, save_dir=None, filename=None):
        filepath = self._get_path(save_dir, filename)

        if not os.path.exists(filepath):
            assert self.dataloaders is not None, ("Must provide dataloaders")

            data = self.extract_and_save(filepath)
        else:
            data = self.load(filepath)
        return data

    def load(self, filepath):
        data = {"train": {}, "validation": {}, "inference": {}}
        with np.load(filepath, allow_pickle=True) as save_data:
            save_data = dict(save_data)
            for key in save_data:
                partition, name = key.split("-")
                data[partition][name] = save_data[key]
        return data

    def extract_and_save(self, filepath):
        data, save_data = {}, {}
        for partition in self.dataloaders.keys():
            data[partition] = self.extract_embeddings(self.dataloaders[partition])

            partition_data = {f"{partition}-{key}": value for key, value in data[partition].items()}
            save_data.update(partition_data)

        np.savez(filepath, **save_data)
        return data

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
        data = {name: [] for name in self.data_struct}
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting embeddings..."):
                batch = self.move_batch_to_device(batch)

                outputs = self.model.features(**batch)
                batch.update(outputs)

                for key in data.keys():
                    value = batch[key].cpu().numpy()
                    data[key].append(value)
        for key in data.keys():
            data[key] = np.concatenate(data[key])
        return data

    def extract_embeddings_siamese(self, dataloader):
        for i in range(len(self.models)):
            self.models[i].eval()
        embeddings = {"siam": [], "ref": [], "coat": [], "conv": []}
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting embeddings..."):
                batch = self.move_batch_to_device(batch)

                quad = {}
                for i in range(len(self.models)):
                    if self.siamese[i]:
                        outputs = self.models[i].test_forward(**batch)
                        emb1 = outputs["r_emb"].cpu().numpy()
                        emb2 = outputs["s_emb"].cpu().numpy()
                        quad["siam"] = emb1
                        quad["ref"] = emb2
                    else:
                        outputs = self.models[i].features(**batch)["emb"]
                        outputs = outputs.mean(axis=(2, 3))
                        if i == 2:
                            quad["coat"] = outputs.cpu().numpy()
                        else: 
                            quad["conv"] = outputs.cpu().numpy()

                for key in quad.keys():
                    embeddings[key].append(quad[key])
                labels.append(batch['labels'].cpu().numpy())
            for key in quad.keys():
                embeddings[key] = np.concatenate(embeddings[key])
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
    
    def _get_path(self, save_dir=None, filename=None):
        if save_dir is None:
            save_dir = ROOT_PATH / "data" / "embeddings"
        os.makedirs(save_dir, exist_ok=True)
        if filename is None:
            filename = f"{self.model.name}_{self.data_mode}.npz"
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