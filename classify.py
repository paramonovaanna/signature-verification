import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_inference_dataloaders
from src.classifier import DistanceClassifier
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="classification")
def main(config):
    set_random_seed(config.classifier.seed)

    if config.classifier.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.classifier.device

    dataloaders, batch_transforms = get_inference_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model._model_).to(device)
    print(model)

    num_users = config.test.users[1] - config.test.users[0] + 1

    classifier = DistanceClassifier(
        model=model,
        config=config.classifier,
        device=device,
        dataloader=dataloaders["inference"],
        batch_transforms=batch_transforms,
        num_users=num_users,
        num_steps=config.classifier.num_steps,
    )
    classifier.extract_embeddings()

    m_values = range(config.classifier.reference_samples[0], config.classifier.reference_samples[1])
    for m in m_values:
        logs = classifier.classify(m)
        print(f"    reference_samples: {m}")
        for key, value in logs.items():
            print(f"    {key:15s}: {value}")


if __name__ == "__main__":
    main()
