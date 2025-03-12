from torch import nn

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class ConvNeXt(nn.Module):

    def __init__(self, base_model):
        super().__init__()
        # Change first layer to accept grayscale images
        self.model = base_model
        self.model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))

        n_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(n_features, 2)

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


class ConvNeXt_T(ConvNeXt):

    def __init__(self, use_pretrained=True):

        weights = None
        if use_pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)

        super().__init__(model)


class ConvNeXt_S(ConvNeXt):

    def __init__(self, use_pretrained=True):

        weights = None
        if use_pretrained:
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = convnext_small(weights=weights)

        super().__init__(model)


class ConvNeXt_B(ConvNeXt):

    def __init__(self, use_pretrained=True):

        weights = None
        if use_pretrained:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=weights)

        super().__init__(model)
