import torch
from torch import nn


class BCEWithLogitsLoss(nn.Module):
    """
    Wrapper over PyTorch BCEWithLogitsLoss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(logits, labels.float())}
