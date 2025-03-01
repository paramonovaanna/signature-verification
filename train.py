import hydra
from hydra.utils import instantiate
import torch

from src.utils.init_utils import set_random_seed
from src.datasets.data_utils import get_dataloaders


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    set_random_seed(config.trainer.seed)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    model = instantiate(config.model._model_)

    dataloaders, batch_transforms = get_dataloaders(config, device)


if __name__ == "__main__":
    main()