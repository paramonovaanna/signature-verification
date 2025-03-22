import os
import numpy as np
from typing import Tuple
from src.datasets.base_dataset import BaseDataset

from src.utils.io_utils import ROOT_PATH

import functools

from src.preprocessor.preprocess_utils import preprocess_signature

from tqdm import tqdm

class HTCSigNetPreprocessor:

    def __init__(
        self,
        canvas_size: Tuple[int, int],
        img_size: Tuple[int, int],
        input_size: Tuple[int, int],
    ):
        """
        Initialize the dataset preprocessor.
        
        Args:
            canvas_size (Tuple[int, int]): Size of the canvas for signature placement
            img_size (Tuple[int, int]): Size of the signature image
            input_size (Tuple[int, int]): Size of the input image for the model
            dataset (BaseDataset): Dataset instance to preprocess

        Return:
            images, labels - np.ndarray
        """
        self.canvas_size = canvas_size
        self.img_size = img_size
        self.input_size = input_size

    def __call__(self, dataset: BaseDataset):
        # Create preprocess directory if it doesn't exist
        preprocess_dir = ROOT_PATH / "data" / "preprocessed"
        os.makedirs(preprocess_dir, exist_ok=True)

        self.preprocessed_file = preprocess_dir / f"{dataset.__class__.__name__}_{self.canvas_size[0]}_{self.canvas_size[1]}_index.npz"
        
        if not os.path.exists(self.preprocessed_file):
            print("Generating preprocessed dataset...")
            images, labels = self.preprocess_dataset(dataset)
        else:
            images, labels = self.load_preprocessed_data()
        
        return images, labels

    def load_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed data from .npz file.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Preprocessed images and labels
        """
        data = np.load(self.preprocessed_file)
        return data["images"], data["labels"] 


    def preprocess_dataset(self, dataset) -> None:
        """
        Preprocess the dataset and save it to a .npz file.
        """
        preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=self.canvas_size,
                                      img_size=self.img_size,
                                      input_size=self.img_size)  # Don't crop it now

        processed = self._preprocess_dataset_images(preprocess_fn, dataset)
        images, labels = processed

        np.savez(self.preprocessed_file,
                images=images,
                labels=labels)
        
        return images, labels
    

    def _preprocess_dataset_images(self, preprocess_fn, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
        """ Process the signature images from a dataset, returning numpy arrays.

        Parameters
        ----------
        preprocess_fn : function (image) -> image
            A function that takes as input a signature image, preprocess-it and return a new image

        Returns
        -------
        images : np.ndarray (N x 1 x H x W) are N grayscale signature images of size H x W
        labels : np.ndarray (N) indicating wether the signature is a forgery
        """
        # Pre-allocate an array X to hold all signatures. We do so because
        # the alternative of concatenating several arrays in the end takes
        # a lot of memory, which can be problematic when using large image sizes
        H, W = self.img_size
        signatures = len(dataset)

        images = np.empty((signatures, H, W), dtype=np.uint8)
        labels = np.empty(signatures, dtype=np.int32)

        for i in tqdm(range(signatures)):
            img, label = dataset.load_img(i, numpy=True)
            preprocessed_img = preprocess_fn(img)

            images[i] = preprocessed_img
            labels[i] = label

        return images, labels