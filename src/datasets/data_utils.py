from itertools import repeat

from PIL import Image

from skimage.io import imread
from skimage import img_as_ubyte


def load_pil(path):
    """
    Load img from disk.

    Args:
        path (str): path to the object.
    Returns:
        img (Tensor):
    """
    img = Image.open(path).convert("L")
    return img
    
def load_numpy(path):
    """
    Load img from disk.

    Args:
        path (str): path to the object.
    Returns:
        img (np.ndarray):
    """
    img = imread(path, as_gray=True)
    return img_as_ubyte(img)

def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    if not batch_transforms:
        return
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)
