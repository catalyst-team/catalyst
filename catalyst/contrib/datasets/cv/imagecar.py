import requests
import cv2

from torch.utils.data import Dataset
from pathlib import Path
from catalyst.contrib.datasets.functional import _extract_archive

DATASET_IDX = '1GSmgsMEs2TE2x__wThw6dMOPfVUgkBZx'


def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


class Car(Dataset):
    def __init__(self, root: str, train: bool = False, download: bool = False, transform=None):
        if download:
            # ToDo check exists
            download_file_from_google_drive(DATASET_IDX, f'{root}/car.zip')
            _extract_archive(f'{root}/car.zip', f'{root}/', True)
        root = Path(root) / 'car'
        split = 'train' if train else 'test'
        mask_path = root / f"{split}_masks"
        image_path = root / f"{split}_images"
        self.image_paths = sorted(image_path.glob("*.png"))
        self.mask_paths = sorted(mask_path.glob("*.gif"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        result = {'image': cv2.imread(image_path),
                  'mask': cv2.imread(mask_path, 2)}
        if self.transforms is not None:
            result = self.transforms(**result)
        return result
