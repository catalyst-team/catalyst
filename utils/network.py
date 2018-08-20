import os
import io
from multiprocessing import Pool
import requests


# @TODO: make it script
OUT_DIR = None


def request_file(url, for_path):
    i = requests.get(url)
    if i.status_code == requests.codes.ok:
        with io.open(for_path, "wb") as file:
            file.write(i.content)
    else:
        return False


def download_file(row):
    if isinstance(row, str):
        url = str(row)
        name = url.rsplit("/", 1)[-1]
    if not url.startswith("http://"):
        url = "http://" + url
    name = os.path.join(OUT_DIR, name)
    request_file(url, name)
    return name


def download_files(files):
    with Pool() as p:
        result = p.map(download_file, files)
    return result
