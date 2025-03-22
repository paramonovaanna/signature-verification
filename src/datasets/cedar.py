import kagglehub
import os
import shutil

import random

from tqdm import tqdm

from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.datasets.base_dataset import BaseDataset


class CEDAR(BaseDataset):

    def __init__(self, path_download, index_dir, *args, **kwargs):

        if path_download is None:
            path_download = ROOT_PATH / "data"
        path_download = Path(path_download)
        self.dataset_path = path_download / "CEDAR"
        if not self.dataset_path.exists():
            self.download_dataset(path_download)

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "indexes" / "cedar"
        index_dir = Path(index_dir)
        index_path = index_dir / f"index.json"

        if index_path.exists():
            self._index = read_json(str(index_path))
        else:
            self._index = self._generate_index(index_dir, str(index_path))

        super().__init__(self._index, *args, **kwargs)

    def download_dataset(self, path_download):
        path = kagglehub.dataset_download("shreelakshmigp/cedardataset")
        if path_download == path:
            return

        print("Extracting files into...", path_download)
        # Ensure the custom directory exists
        os.makedirs(path_download, exist_ok=True)
        # Move downloaded dataset to the custom path
        shutil.move(os.path.join(path, "signatures"), path_download)
        os.rename(path_download / "signatures", path_download / "CEDAR")

    def _generate_index(self, index_dir, index_path):
        '''
        Returns index (list of dicts) with genuine_num genuine signatures 
        and forged_num forged signatures of each person
        '''

        index = []
        index_dir.mkdir(exist_ok=True, parents=True)

        genuine_dir = self.dataset_path / "full_org"
        files = os.listdir(genuine_dir)
        print("Parsing genuine signatures into index...")
        for file in tqdm(files):
            _, extension = os.path.splitext(file)
            if extension != ".png":
                continue
            index.append({
                'path': str(genuine_dir / file),
                'label': 1  # Genuine signatures labeled as 1
            })

        genuine_dir = self.dataset_path / "full_forg"
        files = os.listdir(genuine_dir)
        print("Parsing forged signatures into index...")
        for file in tqdm(files):
            _, extension = os.path.splitext(file)
            if extension != ".png":
                continue
            index.append({
                'path': str(genuine_dir / file),
                'label': 0  # Forged signatures labeled as 1
            })
        
        write_json(index, index_path)
        return index


