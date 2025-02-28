import kagglehub
import os
import shutil

import random

from tqdm import tqdm

from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.datasets.basedataset import BaseDataset


class UTSig(BaseDataset):

    def __init__(self, genuine_num=2, skilled_num=3, opposite_num=0, simple_num=0, *args, **kwargs):
        self.genuine_num = min(genuine_num, 27)
        self.forged_num = {"skilled": min(skilled_num, 6), "opposite hand": min(opposite_num, 3), 
                           "simple": min(simple_num, 36)}

        self.dataset_path = ROOT_PATH / "data" / "UTSig"
        if not self.dataset_path.exists():
            self.download_dataset()

        index_name = f"index_{genuine_num}_{skilled_num}.json"
        index_path = ROOT_PATH / "data" / "UTSig" / "indexes"/ index_name

        if index_path.exists():
            self._index = read_json(str(index_path))
        else:
            self._index = self._generate_index(index_name)

        super().__init__(self._index, *args, **kwargs)

    def download_dataset(self):
        path = kagglehub.dataset_download("sinjinir1999/utsignature-verification")
        custom_path = "./data"
        # Ensure the custom directory exists
        os.makedirs(custom_path, exist_ok=True)
        # Move downloaded dataset to the custom path
        shutil.move(os.path.join(path, "UTSig"), custom_path)

    def _generate_index(self, name):
        '''
        Returns index (list of dicts) with genuine_num genuine signatures 
        and forged_num forged signatures of each person
        '''

        index = []
        path = ROOT_PATH / "data" / "UTSig" / "indexes"
        path.mkdir(exist_ok=True, parents=True)

        genuine_dir = self.dataset_path / "genuine"
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
        write_json(index, str(path / name))
        return index


