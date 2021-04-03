import requests
import cv2

from typing import Optional, Callable
from torch.utils.data import Dataset
from pathlib import Path
from catalyst.contrib.datasets.functional import _extract_archive

DATASET_IDX = '1lq6wOcxtIR3LnIARvlIBZBwJzL7h0FYc'


def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class CarvanaOneCarDataset(Dataset):
    """
    The dataset contains images of cars and the corresponding binary masks for them
    """
    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = False,
                 transforms: Optional[Callable] = None):
        """

        Args:
            root: str: root directory of dataset where
            ``CarvanaOneCarDataset/`` exist.
            train: (bool, optional): If True, creates dataset from
            training part, otherwise from test part
            download: (bool, optional): If true, downloads the dataset from
            the internet and puts it in root directory. If dataset
            is already downloaded, it is not downloaded again.
            transforms: (callable, optional): A function/transform that
            takes in an image and returns a transformed version.

        Examples:
            >>> from catalyst.contrib.datasets import CarvanaOneCarDataset
            >>> dataset = CarvanaOneCarDataset(root='./',
            >>>                                train=True,
            >>>                                download=True,
            >>>                                transforms=None)
            >>> image = dataset[0]['image']
            >>> mask = dataset[0]['mask']
        """
        directory = Path(root) / 'CarvanaOneCarDataset'
        if download and not directory.exists():
            _download_file_from_google_drive(DATASET_IDX, f'{root}/CarvanaOneCarDataset.zip')
            _extract_archive(f'{root}/CarvanaOneCarDataset.zip', f'{root}/', True)
        if not directory.exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        split = 'train' if train else 'test'
        mask_path = directory / f"{split}_masks"
        image_path = directory / f"{split}_images"
        self.image_paths = sorted(image_path.glob("*.jpg"))
        self.mask_paths = sorted(mask_path.glob("*.png"))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])
        result = {'image': cv2.imread(image_path),
                  'mask': cv2.imread(mask_path, 2)}
        if self.transforms is not None:
            result = self.transforms(**result)
        return result
