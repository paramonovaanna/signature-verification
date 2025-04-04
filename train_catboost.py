import torch
import os
import numpy as np

import hydra 
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.io_utils import ROOT_PATH

from src.utils.init_utils import set_random_seed

from src.datasets import DataLoaderFactory
from src.embeddings import EmbeddingsExtractor
from src.models import SiameseNetwork
from src.models import CBClassifier

@hydra.main(version_base=None, config_path="src/configs", config_name="catboost")
def main(config):
    set_random_seed(config.seed)

    if config.embeddings.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.embeddings.device

    model = instantiate(config.model._model_).to(device)
    if config.mode == "siamese":
        model = SiameseNetwork(model).to(device)

    dataloader_factory = DataLoaderFactory(config=config, 
        device=device, 
        train_config=config.train_data, 
        test_config=config.test_data
    )
    dataloaders, batch_transforms = dataloader_factory.get_dataloaders()
    inference_dataloaders, _ = dataloader_factory.get_inference_dataloaders()
    dataloaders.update(inference_dataloaders)

    data_modes = {"train": config.train_data.modes[0], 
        "validation": config.train_data.modes[1], 
        "inference": config.test_data.mode
    }

    emb_extractor = EmbeddingsExtractor(config.embeddings,
        device=device,
        dataloaders=dataloaders, 
        model=model,
        data_modes=data_modes
    )
    data = emb_extractor.extract()

    metrics = instantiate(config.metrics)

    classifier_config = OmegaConf.to_container(config.catboost)

    classifier = CBClassifier(data=data, metrics=metrics, **classifier_config)

    logs = classifier.classify()
    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()