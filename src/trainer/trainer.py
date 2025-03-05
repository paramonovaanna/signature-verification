from src.trainer.base_trainer import BaseTrainer
from src.metrics.metric_tracker import MetricTracker
import torch


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

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        outputs["logits"] = outputs["logits"].squeeze(1)
        batch.update(outputs)

        if self.is_train:
            self.writer.add_scalar("logits/mean", batch["logits"].mean().item())
            self.writer.add_scalar("logits/std", batch["logits"].std().item())
            self.writer.add_scalar("logits/min", batch["logits"].min().item())
            self.writer.add_scalar("logits/max", batch["logits"].max().item())

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
            img = batch["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("image", img)
            
            # Log predictions and labels
            probs = torch.sigmoid(batch["logits"])
            predictions = (probs > 0.5).float()
            
            self.writer.add_scalar("predictions/mean_prob", probs.mean().item())
            self.writer.add_scalar("predictions/std_prob", probs.std().item())
            self.writer.add_scalar("predictions/min_prob", probs.min().item())
            self.writer.add_scalar("predictions/max_prob", probs.max().item())
            self.writer.add_scalar("predictions/accuracy", (predictions == batch["labels"]).float().mean().item())
            
            pred_classes = predictions.cpu().numpy()
            true_classes = batch["labels"].cpu().numpy()
            self.writer.add_scalar("predictions/pred_class_0", (pred_classes == 0).mean())
            self.writer.add_scalar("predictions/pred_class_1", (pred_classes == 1).mean())
            self.writer.add_scalar("predictions/true_class_0", (true_classes == 0).mean())
            self.writer.add_scalar("predictions/true_class_1", (true_classes == 1).mean())

            if self.lr_scheduler is not None:
                self.writer.add_scalar("learning_rate", self.lr_scheduler.get_last_lr()[0])
        else:
            img = batch["img"][0].detach().cpu().numpy().transpose(1, 2, 0)
            self.writer.add_image("image", img)
            
            probs = torch.sigmoid(batch["logits"])
            predictions = (probs > 0.5).float()
            self.writer.add_scalar("test/pred_class_0", (predictions == 0).float().mean().item())
            self.writer.add_scalar("test/pred_class_1", (predictions == 1).float().mean().item())


