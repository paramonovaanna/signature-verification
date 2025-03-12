from src.datasets.base_downloader import BaseDownloader
from tqdm import tqdm
import os

from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json, write_json

class GPDSSynthetic(BaseDownloader):

    GENUINE_EACH, FORGED_EACH = 24, 30

    def __init__(self, path_download, index_dir, *args, **kwargs):
        
        if path_download is None:
            path_download = ROOT_PATH / "data"

        self.dataset_path = Path(path_download) / "GPDS_Synthetic"
        assert self.dataset_path.exists(), ("GPDS Synthetic Signature database cannot be downloaded from the internet. \
                                            For more information see https://gpds.ulpgc.es/downloadnew/download.htm")
        if index_dir is None:
            index_dir = ROOT_PATH / "data" / "indexes"
        index_dir = Path(index_dir)
        index_path = index_dir / "index.json"
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
        print("Parsing signatures into index...")
        for i in tqdm(range(len(subdirs))):
            person_path = self.dataset_path / subdirs[i]
            if not os.path.isdir(person_path):
                continue

            files = os.listdir(person_path)
            for j in range(GPDSSynthetic.GENUINE_EACH):
                index.append({
                    'path': str(person_path / files[j]),
                    'label': 1
                })

            for j in range(GPDSSynthetic.GENUINE_EACH, GPDSSynthetic.GENUINE_EACH + GPDSSynthetic.FORGED_EACH):
                index.append({
                    'path': str(person_path / files[j]),
                    'label': 0
                })

        write_json(index, index_path)
        return index

