import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Dict, Any, Tuple
import logging

import numpy as np

from src.trainer.base_trainer import BaseTrainer

class SiameseTrainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity_fn = nn.CosineSimilarity(dim=1, eps=1e-8)

    def process_batch(self, batch, metrics):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["test"]["batch"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        outputs["a_emb"] = outputs["a_emb"].squeeze(1)
        outputs["p_emb"] = outputs["p_emb"].squeeze(1)
        outputs["n_emb"] = outputs["n_emb"].squeeze(1)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        batch["loss"].backward()  # sum of all losses is always called loss
        self._clip_grad_norm()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch


    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            all_probs, all_labels = [], []

            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):

                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)

                outputs = self.model.test_forward(**batch)
                outputs["r_emb"] = outputs["r_emb"].squeeze(1)
                outputs["s_emb"] = outputs["s_emb"].squeeze(1)
                batch.update(outputs)

                similarities = self.similarity_fn(outputs["r_emb"], outputs["s_emb"])
                genuine_prob = (similarities.unsqueeze(1) + 1) / 2
                forged_prob = 1 - genuine_prob
                probs = torch.cat([forged_prob, genuine_prob], dim=1).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(batch["labels"].cpu().numpy())
        
            all_probs = np.concatenate(all_probs)
            all_labels = np.concatenate(all_labels)
            self._calculate_epoch_metrics(all_probs, all_labels, self.evaluation_metrics)

            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

            return self.evaluation_metrics.result()

    def _calculate_epoch_metrics(self, all_probs, all_labels, metrics):
        """ In this version: only calculate EER, EER_Accuracy"""
        metric_funcs = self.metrics["test"]["epoch"]

        for met in metric_funcs:
            if met.name == "EER":
                eer, threshold = met(all_probs, all_labels)
                eer_accuracy = met.get_eer_accuracy(all_probs, all_labels, threshold)
                metrics.update("EER", eer)
                metrics.update("EER_Accuracy", eer_accuracy)
                metrics.update("Threshold", threshold)
            else:
                metrics.update(met.name, met(all_logits, all_labels))

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":
            img = batch["anchor"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("anchor_signature", img)
            img = batch["positive"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("pos_signature", img)
            img = batch["negative"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("neg_signature", img)
        else:
            img = batch["reference"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("reference", img)
            img = batch["sig"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("query_signature", img)