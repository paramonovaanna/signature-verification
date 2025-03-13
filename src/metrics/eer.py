import torch

import numpy as np
from sklearn.metrics import roc_curve

from src.metrics.basemetric import BaseMetric


class EER(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Equel Error Rate (EER): FAR = FRR
        """
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            EER (float): calculated metric.
        """
        fpr, tpr, thresholds = roc_curve(labels, logits[:, 1], pos_label=1)
        fnr = 1 - tpr

        eer_index = np.nanargmin(np.abs(fnr - fpr))
    
        eer = (fpr[eer_index] + fnr[eer_index]) / 2
        eer_threshold = thresholds[eer_index]
        
        return eer
