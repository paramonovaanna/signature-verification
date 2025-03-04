from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch import nn


class ConvNeXt_T(nn.Module):

    def __init__(self, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = convnext_tiny(weights=weights)

        n_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(n_features, 1)

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
        return self.model.classifier.parameters()
