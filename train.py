import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.datasets.dataloader_factory import DataLoaderFactory
from src.trainer import Trainer

from src.models import SiameseNetwork

from src.preprocessor import HTCSigNetPreprocessor


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
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
    if config.mode == "siamese":
        model = SiameseNetwork(model).to(device)
    logger.info(model)

    dataloader_factory = DataLoaderFactory(config=config, device=device, train_config=config.data)
    dataloaders, batch_transforms = dataloader_factory.get_dataloaders()

    loss_function = instantiate(config.loss).to(device)
    metrics = instantiate(config.metrics)

    optimizer = instantiate(config.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    trainer = Trainer(
        model=model,
        mode=config.mode,
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
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()

if __name__ == "__main__":
    main()