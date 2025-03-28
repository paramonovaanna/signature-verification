import torch

from src.utils.init_utils import set_random_seed

import hydra 
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders, get_inference_dataloaders

from src.embeddings import EmbeddingsExtractor

from src.utils.io_utils import ROOT_PATH

@hydra.main(version_base=None, config_path="src/configs", config_name="catboost")
def main(config):
    set_random_seed(config.seed)

    if config.device.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device.device

    model1 = instantiate(config.model._model_).to(device)
    #model2 = instantiate(config.model2._model_).to(device)

    dataloaders, batch_transforms = get_dataloaders(config, device)
    test_dataloaders, batch_transforms = get_inference_dataloaders(config, device)
    dataloaders.update(test_dataloaders)

    extractor = EmbeddingsExtractor(
        models=[model1, None], 
        pretrained_paths=config.embeddings.pretrained_paths,
        device=device,
        device_tensors=config.device.device_tensors
    )

    emb, labels = extractor.extract(
        save_dir=config.embeddings.save_dir,
        filename=config.embeddings.filename,
        dataloaders=dataloaders,
    )

    #loss_function = instantiate(config.loss).to(device)
    #metrics = instantiate(config.metrics)

if __name__ == "__main__":
    main()