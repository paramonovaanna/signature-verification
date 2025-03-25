from src.datasets.base_dataset import BaseDataset
from tqdm import tqdm
import os

from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json, write_json

class GPDSSynthetic(BaseDataset):

    def __init__(self, signatures_per_user, path_download, index_dir, *args, **kwargs):

        self.signatures_per_user = signatures_per_user
        
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

