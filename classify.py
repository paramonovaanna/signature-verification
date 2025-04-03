import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets import DataLoaderFactory
from src.embeddings import EmbeddingsExtractor
from src.utils.init_utils import set_random_seed

from src.models import SiameseNetwork

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="classification")
def main(config):
    set_random_seed(config.seed)

    if config.embeddings.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.embeddings.device

    dataloader_factory = DataLoaderFactory(config, device)
    dataloaders, batch_transforms = dataloader_factory.get_inference_dataloaders(config.data.mode)

    # build model architecture, then print to console
    model = instantiate(config.model._model_).to(device)
    if config.mode == "siamese":
        model = SiameseNetwork(model)
    print(model)

    num_users = config.data.users[1] - config.data.users[0] + 1

    emb_extractor = EmbeddingsExtractor(config.embeddings,
        device=device,
        dataloaders=dataloaders, 
        model=model,
        data_mode=config.data.mode,
    )
    data = emb_extractor.extract()["inference"]
    if data.get("emb", None) is not None:
        embeddings = data["emb"]
    else:
        embeddings = data["s_emb"]

    classifier = instantiate(config.classifier, 
        num_users=num_users,
        user2emb=data["user"],
        labels=data["labels"],
        embeddings=embeddings,
        references=data.get("r_emb", None)
    )

    logs = classifier.classify()
    for log in logs:
        for key, value in log.items():
            print(f"    {key:15s}: {value}")


if __name__ == "__main__":
    main()
