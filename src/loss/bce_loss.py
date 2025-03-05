import torch
from torch import nn


class BCEWithLogitsLoss(nn.Module):
    """
    Wrapper over PyTorch BCEWithLogitsLoss with class weights
    """

    def __init__(self, pos_weight=1.0):
        """
        Args:
            pos_weight (float): weight for positive class (class 1)
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.loss = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        if self.loss is None:
            self.loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([self.pos_weight], device=logits.device)
            )
        return {"loss": self.loss(logits, labels.float())}
