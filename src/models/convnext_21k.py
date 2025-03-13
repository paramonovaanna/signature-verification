import timm

from torch import nn

class ConvNeXt_21k(nn.Module):

    def __init__(self, model_name, use_pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=use_pretrained)

        # Change first layer to accept grayscale images
        self.model.stem[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))

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