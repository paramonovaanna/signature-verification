from src.datasets.base_dataset import BaseDataset
from tqdm import tqdm
import os

from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json, write_json

class GPDSSynthetic(BaseDataset):

    def __init__(self, first_user, last_user, path_download, index_dir, *args, **kwargs):

        assert 1 <= first_user <= last_user <= 4000
        
        if path_download is None:
            path_download = ROOT_PATH / "data"
        self.dataset_path = Path(path_download) / "GPDS_Synthetic"
        assert self.dataset_path.exists(), ("GPDS Synthetic Signature database cannot be downloaded from the internet. \
                                            For more information see https://gpds.ulpgc.es/downloadnew/download.htm")

        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "indexes"
        index_dir = Path(index_dir)
        index_path = index_dir / f"index.json"
    
        if index_path.exists():
            self._index = read_json(str(index_path))
        else:
            self._index = self._generate_index(index_dir, str(index_path))

        self._index = self._limit_index(first_user, last_user)

        super().__init__(self._index, *args, **kwargs)

    def _generate_index(self, index_dir, index_path):
        '''
        Returns index (list of dicts) with genuine signatures labeled as 1
        and forged_num labeled as 0
        '''

        index = []
        index_dir.mkdir(exist_ok=True, parents=True)

        subdirs = os.listdir(self.dataset_path)
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
    
    def _extract_user_no(self, path):
        filename = os.path.basename(path)
        number = int(filename.split("-")[1])
        return number
    
    def _limit_index(self, first_user, last_user):
        index = sorted(self._index, key=lambda x: self._extract_user_no(x["path"]))
        start = (first_user - 1) * 54
        end = last_user * 54
        return index[start:end]


