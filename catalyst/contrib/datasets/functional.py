import codecs
import gzip
import hashlib
import lzma
import os
import tarfile
import zipfile

import numpy as np
import torch
from torch.utils.model_zoo import tqdm


def _gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def _calculate_md5(fpath, chunk_size=1024 * 1024):  # noqa: WPS404
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _check_md5(fpath, md5, **kwargs):
    return md5 == _calculate_md5(fpath, **kwargs)


def _check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return _check_md5(fpath, md5)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url: URL to download file from
        root: Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
            If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download.
            If None, do not check

    Raises:
        IOError: if failed to download url
        RuntimeError: if file not found or corrupted
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if _check_integrity(fpath, md5):
        print("Using downloaded and verified file: " + fpath)
    else:  # download the file
        try:
            print("Downloading " + url + " to " + fpath)
            urllib.request.urlretrieve(url, fpath, reporthook=_gen_bar_updater())
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == "https":
                url = url.replace("https:", "http:")
                print(
                    "Failed download. Trying https -> http instead."
                    " Downloading " + url + " to " + fpath
                )
                urllib.request.urlretrieve(url, fpath, reporthook=_gen_bar_updater())
            else:
                raise e
        # check integrity of downloaded file
        if not _check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def _extract_archive(from_path, to_path=None, remove_finished=False):
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if from_path.endswith(".tar"):
        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".tar.gz") or from_path.endswith(".tgz"):
        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".tar.xz"):
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif from_path.endswith(".gz"):
        root, _ = os.path.splitext(os.path.basename(from_path))
        to_path = os.path.join(to_path, root)
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:  # noqa: WPS316
            out_f.write(zip_f.read())
    elif from_path.endswith(".zip"):
        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError(f"Extraction of {from_path} not supported")

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url, download_root, extract_root=None, filename=None, md5=None, remove_finished=False,
):
    """@TODO: Docs. Contribution is welcome."""
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print(f"Extracting {archive} to {extract_root}")
    _extract_archive(archive, extract_root, remove_finished)


def _get_int(b):
    """@TODO: Docs. Contribution is welcome."""
    return int(codecs.encode(b, "hex"), 16)


def _open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
    Decompression occurs when argument `path` is a string
    and ends with '.gz' or '.xz'.

    Args:
        path: path

    Returns:
        file
    """
    if not isinstance(path, torch._six.string_classes):  # noqa: WPS437
        return path
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    if path.endswith(".xz"):
        return lzma.open(path, "rb")
    return open(path, "rb")


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format."""
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, "typemap"):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype(">i2"), "i2"),
            12: (torch.int32, np.dtype(">i4"), "i4"),
            13: (torch.float32, np.dtype(">f4"), "f4"),
            14: (torch.float64, np.dtype(">f8"), "f8"),
        }
    # read
    with _open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = _get_int(data[0:4])  # noqa: WPS349
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [_get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


__all__ = ["download_and_extract_archive", "download_url", "read_sn3_pascalvincent_tensor"]
