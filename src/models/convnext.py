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
        
        # Change first layer to accept grayscale images
        self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))

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
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        return self.model.classifier.parameters()
