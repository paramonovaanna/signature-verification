import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_inference_dataloaders
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

from src.classifiers import CosineClassifier


@hydra.main(version_base=None, config_path="src/configs", config_name="classification")
def main(config):
    set_random_seed(config.seed)

    if config.classifier.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.classifier.device

    dataloaders, batch_transforms = get_inference_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model._model_).to(device)
    print(model)

    num_users = config.test.users[1] - config.test.users[0] + 1

    classifier = instantiate(config.classifier,
        model=model,
        dataloader=dataloaders["inference"],
        batch_transforms=batch_transforms,
        num_users=num_users,
        device=device
    )
    classifier.extract_embeddings()

    logs = classifier.classify()
    for log in logs:
        for key, value in log.items():
            print(f"    {key:15s}: {value}")


if __name__ == "__main__":
    main()
