import torch
from torchvision import transforms

class GrayscaleToRGB(transforms.Lambda):
    def __init__(self):
        super().__init__(lambda x: x.repeat(3, 1, 1)) 