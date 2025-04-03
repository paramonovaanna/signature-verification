import torch
import torch.nn as nn
from tqdm.auto import tqdm

import numpy as np

from src.metrics.metric_tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        mode,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.similarity_fn = nn.CosineSimilarity(dim=1, eps=1e-8) # for siamese networks

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.mode = mode
        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.optim_threshold = None
        self.metrics = metrics
        if self.metrics is not None:
            metrics_names = [m.name for period in self.metrics["inference"].values() for m in period]
            self.evaluation_metrics = MetricTracker(
                "EER_Accuracy",
                "EER_Optim_Accuracy",
                "Threshold", 
                "Optim_Threshold",
                *metrics_names,
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]["batch"]:
                metrics.update(met.name, met(**batch))

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            values, labels = [], []
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                if self.mode == "siamese":
                    outputs = self.model.forward(**batch)
                    outputs["r_emb"] = outputs["r_emb"].squeeze(1)
                    outputs["s_emb"] = outputs["s_emb"].squeeze(1)
                    batch.update(outputs)
                    probs = self._calculate_probs(outputs["r_emb"], outputs["s_emb"])
                    values.append(probs)
                else: 
                    batch = self.process_batch(
                        batch_idx=batch_idx,
                        batch=batch,
                        part=part,
                        metrics=self.evaluation_metrics,
                    )
                    values.append(batch["logits"].cpu().numpy())

                labels.append(batch["labels"].cpu().numpy())

            values = np.concatenate(values)
            labels = np.concatenate(labels)
            self._calculate_epoch_metrics(values, labels, self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _calculate_epoch_metrics(self, values, labels, metrics: MetricTracker):
        """ In this version: only calculate EER, EER_Accuracy"""
        """ We calculate two EER accuracies: using threshold from EER and from the trained model"""
        metric_funcs = self.metrics["inference"]["epoch"]

        for met in metric_funcs:
            if met.name == "EER":
                eer, threshold = met(values, labels)
                
                eer_accuracy = met.get_eer_accuracy(values, labels, threshold)
                metrics.update("EER", eer)
                metrics.update("EER_Accuracy", eer_accuracy)
                metrics.update("Threshold", threshold)
                if self.optim_threshold is not None:
                    eer_optim_accuracy = met.get_eer_accuracy(values, labels, self.optim_threshold)
                    metrics.update("EER_Optim_Accuracy", eer_optim_accuracy)
                    metrics.update("Optim_Threshold", self.optim_threshold)
            else:
                metrics.update(met.name, met(values, labels))
