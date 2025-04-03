from src.trainer.base_trainer import BaseTrainer
from src.metrics.metric_tracker import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
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
        for key in ["logits", "s_emb", "r_emb", "a_emb", "n_emb", "p_emb"]:
            if outputs.get(key, None) is not None:
                outputs[key] = outputs[key].squeeze(1)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _calculate_epoch_metrics(self, values, labels, metrics: MetricTracker):
        """ In this version: only calculate EER, EER_Accuracy"""
        metric_funcs = self.metrics["test"]["epoch"]

        for met in metric_funcs:
            if met.name == "EER":
                eer, threshold = met(values, labels)
                eer_accuracy = met.get_eer_accuracy(values, labels, threshold)
                metrics.update("EER", eer)
                metrics.update("EER_Accuracy", eer_accuracy)
                metrics.update("Threshold", threshold)
            else:
                metrics.update(met.name, met(values, labels))
    
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
        if batch.get("anchor", None) is not None:
            img = batch["anchor"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("anchor", img)
            img = batch["pos"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("positive", img)
            img = batch["neg"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("negative", img)
            
        elif batch.get("ref", None) is not None:
            img = batch["ref"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("reference", img)
            img = batch["sig"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("signature", img)

        else:
            img = batch["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("image", img)

