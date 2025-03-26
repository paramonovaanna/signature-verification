from torch import nn
import torch

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models.convnext import LayerNorm2d

class ConvNeXt(nn.Module):
    """
    N_LAYERS = 8
    """

    def __init__(self, base_model, weights=None):
        super().__init__()
        self.model = base_model
        
        if weights is not None:
            first_conv_weights = self.model.features[0][0].weight.data
            grayscale_weights = first_conv_weights.mean(dim=1, keepdim=True)
        
        # Change first layer to accept grayscale images
        in_chans = 1  # grayscale
        first_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(in_chans, 
                                             first_conv.out_channels, 
                                             kernel_size=first_conv.kernel_size,
                                             stride=first_conv.stride,
                                             padding=first_conv.padding)
        
        if weights is not None:
            self.model.features[0][0].weight.data = grayscale_weights

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

    def features(self, img, **batch):
        return {"emb": self.model.forward_features(img)}
    
    def freeze_layers(self, num_layers=None):
        # замораживать слои, кроме классификатора
        if num_layers is None:
            return
        assert (num_layers - 1) < len(self.model.features)

        for param in self.model.features[:num_layers].parameters():
            param.requires_grad = False

class ConvNeXt_T(ConvNeXt):
    def __init__(self, use_pretrained=True):
        weights = None
        if use_pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)  # Load with weights first
        super().__init__(model, weights=weights if use_pretrained else None)

class ConvNeXt_S(ConvNeXt):
    def __init__(self, use_pretrained=True):
        weights = None
        if use_pretrained:
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = convnext_small(weights=weights)  # Load with weights first
        super().__init__(model, weights=weights if use_pretrained else None)

class ConvNeXt_B(ConvNeXt):
    def __init__(self, use_pretrained=True):
        weights = None
        if use_pretrained:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=weights)  # Load with weights first
        super().__init__(model, weights=weights if use_pretrained else None)
