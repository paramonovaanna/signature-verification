import torch

from src.metrics.basemetric import BaseMetric


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Accuracy Metric for binary classification with BCEWithLogitsLoss
        """
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic for binary classification.
        Applies sigmoid to logits and rounds to get binary predictions.

        Args:
            logits (Tensor): model output predictions (logits).
            labels (Tensor): ground-truth labels (0 or 1).
        Returns:
            accuracy (float): calculated metric.
        """
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).float()
        return (predictions == labels).mean(dtype=torch.float32)