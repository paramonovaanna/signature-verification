from src.datasets.base_dataset import BaseDataset
from tqdm import tqdm
import os

import random

from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json, write_json

class GPDSSynthetic(BaseDataset):

    def __init__(self, users, path_download, index_dir, *args, **kwargs):

        assert 1 <= users[0] <= users[1] <= 4000
        
        if path_download is None:
            path_download = ROOT_PATH / "data"
        self.dataset_path = Path(path_download) / "GPDS_Synthetic"
        assert self.dataset_path.exists(), ("GPDS Synthetic Signature database cannot be downloaded from the internet. \
                                            For more information see https://gpds.ulpgc.es/downloadnew/download.htm")

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "indexes"
        index_dir = Path(index_dir)
        index_path = index_dir / f"gpds_index.json"
    
        if index_path.exists():
            self._index = read_json(str(index_path))
        else:
            self._index = self._generate_index(index_dir, str(index_path))

        self._index = self._limit_index(users[0], users[1])

        super().__init__(self._index, *args, **kwargs)

    def _generate_index(self, index_dir, index_path):
        '''
        Returns index (list of dicts) with genuine signatures labeled as 1
        and forged_num labeled as 0
        '''

        index = []
        index_dir.mkdir(exist_ok=True, parents=True)

        subdirs = os.listdir(self.dataset_path)
        subdirs = sorted(subdirs, key=lambda x: int(os.path.basename(x)))
        print("Parsing signatures into index...")
        for i in tqdm(range(len(subdirs))):
            person_path = self.dataset_path / subdirs[i]
            if not os.path.isdir(person_path):
                continue

            files = os.listdir(person_path)
            for filename in files:
                name, extension = os.path.splitext(filename)
                if extension == "mat":
                    continue

                if name.startswith("c-"):
                    index.append({
                        'path': str(person_path / filename),
                        'label': 1
                    })

                if name.startswith("cf-"):
                    index.append({
                        'path': str(person_path / filename),
                        'label': 0
                    })

        write_json(index, index_path)
        return index

    def split_dataset(self, split):
        signers = self.__len__() // 54
        users_range = list(range(0, signers))
        train_len = int(signers * split)

        print(train_len, signers)

        train_set = random.sample(users_range, train_len)
        test_set = [no for no in users_range if no not in train_set]

        train_index, test_index = [], []
        for no in users_range:
            first_sig, last_sig = no * 54, (no + 1) * 54
            if no in train_set:
                train_index.extend(self._index[first_sig:last_sig])
            else:
                test_index.extend(self._index[first_sig:last_sig])

        print(len(train_index), len(test_index))
        return BaseDataset(train_index), BaseDataset(test_index)


    def _extract_user_no(self, path):
        filename = os.path.basename(path)
        number = int(filename.split("-")[1])
        return number
    
    def _limit_index(self, first_user, last_user):
        index = sorted(self._index, key=lambda x: self._extract_user_no(x["path"]))
        start = (first_user - 1) * 54
        end = last_user * 54
        return index[start:end]


