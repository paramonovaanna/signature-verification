import os
import numpy as np
from typing import Tuple
from src.datasets.base_dataset import BaseDataset

from src.utils.io_utils import ROOT_PATH

import functools

from src.preprocessor.preprocess_utils import preprocess_signature

from src.datasets import PreprocessedDataset

from tqdm import tqdm

import torch

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
        
        self.preprocessed_file = preprocess_dir / f"{dataset.name}_index.npz"

        if not os.path.exists(self.preprocessed_file):
            print("Generating preprocessed dataset...")
            images, labels, user_mapping = self.preprocess_dataset(dataset)
        else:
            images, labels, user_mapping = self.load_preprocessed_dataset(self.preprocessed_file)
        
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)
        return PreprocessedDataset(images, labels, user_mapping)
    
    def load_preprocessed_dataset(self, path: str):
        """ Loads a dataset that was pre-processed in a numpy format

        Parameters
        ----------
        path : str
            The path to a .npz file, containing attributes "x", "y", "yforg",
            "usermapping"

        Returns
        -------
        x : np.ndarray (N x 1 x H x W) are N grayscale signature images of size H x W
        y : np.ndarray (N) indicating the user that wrote the signature
        yforg : np.ndarray (N) indicating wether the signature is a forgery
        user_mapping: dict, mapping the indexes in "y" with the original user
                    numbers from the dataset
        -------

        """
        with np.load(path, allow_pickle=True) as data:
            images, labels = data['images'], data['labels']
            user_mapping = data['user_mapping']

        return images, labels, user_mapping


    def preprocess_dataset(self, dataset) -> None:
        """
        Preprocess the dataset and save it to a .npz file.
        """
        preprocess_fn = functools.partial(preprocess_signature,
                                      canvas_size=self.canvas_size,
                                      img_size=self.img_size,
                                      input_size=self.img_size)  # Don't crop it now

        processed = self._preprocess_dataset_images(dataset, preprocess_fn)
        images, labels, user_mapping = processed

        np.savez(self.preprocessed_file,
                images=images,
                labels=labels,
                user_mapping=user_mapping)
        
        return images, labels, user_mapping


    def _preprocess_dataset_images(self, dataset, preprocess_fn):
        """ Process the signature images from a dataset, returning numpy arrays.

        Parameters
        ----------
        dataset : IterableDataset
            The dataset, that knows where the signature files are located
        preprocess_fn : function (image) -> image
            A function that takes as input a signature image, preprocess-it and return a new image
        img_size : tuple (H x W)
            The final size of the images
        subset : slice
            Which users to consider. Either "None" (to consider all users) or a slice(first, last)

        Returns
        -------
        images : np.ndarray (N x 1 x H x W) are N grayscale signature images of size H x W
        user_mapping : np.ndarray (N) indicating the user that wrote the signature
        labels : np.ndarray (N) indicating wether the signature is a forgery
        """
        users = dataset.get_user_list()

        H, W = self.img_size
        max_signatures = len(users) * dataset.signatures_per_user

        images = np.empty((max_signatures, H, W), dtype=np.uint8)
        user_mapping = np.empty(max_signatures, dtype=np.int32)
        labels = np.empty(max_signatures, dtype=np.int32)

        N = 0
        for user in tqdm(users):
            gen_imgs = [preprocess_fn(img) for img in dataset.iter_genuine(user)]
            new_img_count = len(gen_imgs)

            indexes = slice(N, N + new_img_count)
            images[indexes] = gen_imgs
            labels[indexes] = 1
            user_mapping[indexes] = user
            N += new_img_count

            forg_imgs = [preprocess_fn(img) for img in dataset.iter_forged(user)]
            if len(forg_imgs) > 0:
                new_img_count = len(forg_imgs)

                indexes = slice(N, N + new_img_count)
                images[indexes] = forg_imgs
                labels[indexes] = 0
                user_mapping[indexes] = user
                N += new_img_count
        return images, labels, user_mapping