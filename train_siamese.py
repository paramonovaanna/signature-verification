import logging

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.models.siamese_networks import SiameseNetwork
from src.siamese_data.data_utils import get_dataloaders, get_inference_dataloaders
from src.trainer import SiameseTrainer


@hydra.main(version_base=None, config_path="src/configs", config_name="siamese")
def main(config):
    set_random_seed(config.trainer.seed)

    logger = setup_saving_and_logging(config)
    project_config = OmegaConf.to_container(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device
    logger.info(device)

    model = instantiate(config.model._model_).to(device)
    siam_model = SiameseNetwork(model).to(device)
    #logger.info(siam_model)

    dataloaders, batch_transforms = get_dataloaders(config, device)
    inference_dataloaders, batch_transforms = get_inference_dataloaders(config, device)
    #dataloaders["test"] = inference_dataloaders["inference"]

    loss_function = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)

    optimizer = instantiate(config.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    trainer = SiameseTrainer(
        model=siam_model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=config.trainer.get("epoch_len"),
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True)
    )
    trainer.train()

if __name__ == "__main__":
    main()