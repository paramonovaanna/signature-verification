import kagglehub
import os
import shutil

import random

from tqdm import tqdm

from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.datasets.base_downloader import BaseDownloader


class UTSig(BaseDownloader):

    def __init__(self, genuine_num, skilled_num, opposite_num, simple_num, 
                 path_download, index_dir, *args, **kwargs):

        self.genuine_num = min(genuine_num, 27)
        self.forged_num = {"Skilled": min(skilled_num, 6), "Opposite Hand": min(opposite_num, 3), 
                           "Simple": min(simple_num, 36)}

        if path_download is None:
            path_download = ROOT_PATH / "data"
        path_download = Path(path_download)
        self.dataset_path = path_download / "UTSig"
        if not self.dataset_path.exists():
            self.download_dataset(path_download)

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "indexes" / "utsig"
        index_dir = Path(index_dir)
        index_path = index_dir / f"index_{genuine_num}_{skilled_num}.json"

        if index_path.exists():
            self._index = read_json(str(index_path))
        else:
            self._index = self._generate_index(index_dir, str(index_path))

        super().__init__(self._index, *args, **kwargs)

    def download_dataset(self, path_download):
        path = kagglehub.dataset_download("sinjinir1999/utsignature-verification")
        if path_download == path:
            return

        print("Extracting files into...", path_download)
        # Ensure the custom directory exists
        os.makedirs(path_download, exist_ok=True)
        # Move downloaded dataset to the custom path
        shutil.move(os.path.join(path, "UTSig"), path_download)

    def _generate_index(self, index_dir, index_path):
        '''
        Returns index (list of dicts) with genuine_num genuine signatures 
        and forged_num forged signatures of each person
        '''

        index = []
        index_dir.mkdir(exist_ok=True, parents=True)

        genuine_dir = self.dataset_path / "Genuine"
        genuine_subdirs = os.listdir(genuine_dir)
        print("Parsing genuine signatures into index...")
        for i in tqdm(range(len(genuine_subdirs))):
            person_path = genuine_dir / genuine_subdirs[i]
            if not os.path.isdir(person_path):
                continue

            genuine_files = random.sample(sorted(os.listdir(person_path)), self.genuine_num)
            for file in genuine_files:
                index.append({
                    'path': str(person_path / file),
                    'label': 1  # Genuine signatures labeled as 1
                })
        
        for type in self.forged_num.keys():
            if self.forged_num[type] <= 0:
                continue

            forged_dir = self.dataset_path / "Forgery" / type
            forged_subdirs = os.listdir(forged_dir)
            print(f"Parsing {type} forgeries into index...")
            for i in tqdm(range(len(forged_subdirs))):
                person_path = forged_dir / forged_subdirs[i]
                if not os.path.isdir(person_path):
                    continue

                forged_files = random.sample(sorted(os.listdir(person_path)), self.forged_num[type])
                for file in forged_files:
                    index.append({
                        'path': str(person_path / file),
                        'label': 0
                    })
        write_json(index, index_path)
        return index


