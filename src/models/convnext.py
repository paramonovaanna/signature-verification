from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch import nn
import torch.nn.functional as F


class ConvNeXt_T(nn.Module):

    def __init__(self, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = convnext_tiny(weights=weights)

        n_features = self.model.classifier[2].in_features
        self.model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LayerNorm(n_features, eps=1e-06, elementwise_affine=True),
            nn.Dropout(p=0.2),
            nn.Linear(n_features, 1)
        )
        
        nn.init.xavier_uniform_(self.model.classifier[3].weight, gain=1.0)
        nn.init.zeros_(self.model.classifier[3].bias)

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input img.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.model(img)}

    def classifier_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        return self.model.classifier.parameters()
