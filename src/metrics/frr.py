import torch

from src.metrics.basemetric import BaseMetric


class FRR(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        False Rejection Rate (FRR) = False Rejected / Total Genuine
        """
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            FRR (float): calculated metric.
        """
        classes = logits.argmax(dim=-1)
        false_rejects = ((classes == 0) & (labels == 1)).sum().item()
        total_genuine = (labels == 1).sum().item()
        if total_genuine == 0:
            return 0.0
        return false_rejects / total_genuine