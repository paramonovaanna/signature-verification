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