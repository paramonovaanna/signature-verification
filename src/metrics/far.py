import torch

from src.metrics.basemetric import BaseMetric


class FAR(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        False Acceptance Rate (FAR) = False Accepted / Total Forged
        """
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            FAR (float): calculated metric.
        """
        classes = logits.argmax(dim=-1)
        false_accepts = ((classes == 1) & (labels == 0)).sum().item()
        total_forged = (labels == 0).sum().item()
        if total_forged == 0:
            return 0.0
        return false_accepts / total_forged