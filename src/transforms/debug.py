import torch
import logging

logger = logging.getLogger(__name__)

class DebugTransform:
    def __init__(self, name):
        self.name = name
        self.counter = 0

    def __call__(self, img):
        if self.counter < 5:  # Log only first 5 images
            logger.info(f"{self.name} transform stats:")
            logger.info(f"  Shape: {img.shape}")
            logger.info(f"  Min: {img.min():.3f}, Max: {img.max():.3f}")
            logger.info(f"  Mean: {img.mean():.3f}, Std: {img.std():.3f}")
            logger.info(f"  Unique values: {torch.unique(img)}")
            self.counter += 1
        return img 