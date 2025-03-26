import timm

from torch import nn
from torchvision.models.convnext import LayerNorm2d

class ConvNeXt_21k(nn.Module):
    """
    stem: tiny - 1 block, small - 1 block
    stages: tiny - 4 stages (downsample + blocks: 3, 3, 9, 3)
            tiny - 4 stages (downsample + blocks: 3, 3, 19, 3)
            base - 4 stages (downsample + blocks: 3, 3, 27, 3)
    head:
    """

    def __init__(self, model_name, use_pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=use_pretrained)

        if use_pretrained:
            first_conv_weights = self.model.stem[0].weight.data
            grayscale_weights = first_conv_weights.mean(dim=1, keepdim=True)

        in_chans = 1
        first_conv = self.model.stem[0]
        self.model.stem[0] = nn.Conv2d(in_chans, 
                                             first_conv.out_channels, 
                                             kernel_size=first_conv.kernel_size,
                                             stride=first_conv.stride,
                                             padding=first_conv.padding)
        
        if use_pretrained:
            self.model.stem[0].weight.data = grayscale_weights

        # Change classifier to classify into genuine and forged
        n_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features=n_features, out_features=2, bias=True)

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
        # замораживаем все, кроме классификатора

        if num_layers is None:
            return

        assert num_layers < len(self.model.stages) + 1

        for param in self.model.stem.parameters():
            param.requires_grad = False

        for param in self.model.stages[:num_layers - 1].parameters():
            param.requires_grad = False