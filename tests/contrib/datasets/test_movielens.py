import os
import shutil

import pytest

import torch

from catalyst.settings import SETTINGS
from tests import MOVIELENS_ROOT

if SETTINGS.ml_required:
    from catalyst.contrib.datasets import MovieLens


def setup_module():
    """
    Remove the temp folder if exists
    """
    data_path = MOVIELENS_ROOT
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_download():
    """
    Test movielense download
    """
    MovieLens(MOVIELENS_ROOT, download=True)

    filename = "ml-100k"

    # check if data folder exists
    assert os.path.isdir(MOVIELENS_ROOT) is True

    # cehck if class folder exists
    assert os.path.isdir(f"{MOVIELENS_ROOT}/MovieLens") is True

    # check if raw folder exists
    assert os.path.isdir(f"{MOVIELENS_ROOT}/MovieLens/raw") is True

    # check if processed folder exists
    assert os.path.isdir(f"{MOVIELENS_ROOT}/MovieLens/processed") is True

    # check if raw folder exists
    assert os.path.isdir(f"{MOVIELENS_ROOT}/MovieLens/raw") is True

    # check some random file from MovieLens
    assert os.path.isfile(f"{MOVIELENS_ROOT}/MovieLens/raw/{filename}/u.info") is True

    # check Readme file from MovieLens
    assert os.path.isfile(f"{MOVIELENS_ROOT}/MovieLens/raw/{filename}/README") is True

    # check if data file is not Nulll
    assert os.path.getsize(f"{MOVIELENS_ROOT}/MovieLens/raw/{filename}/u.data") > 0

    # check if info file is not Nulll
    assert os.path.getsize(f"{MOVIELENS_ROOT}/MovieLens/raw/{filename}/u.info") > 0


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_reading():
    """
    Test reading of the movielens dataset
    """
    training_file = "training.pt"
    test_file = "test.pt"
    processed_folder = f"{MOVIELENS_ROOT}/MovieLens/processed"

    assert torch.load(os.path.join(processed_folder, test_file)).size() == torch.Size(
        [943, 1682]
    )
    assert torch.load(
        os.path.join(processed_folder, training_file)
    ).size() == torch.Size([943, 1682])


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_minimal_ranking():
    """
    Tets retrieveing the minimal ranking
    """
    train_data_loader_min_two = MovieLens(MOVIELENS_ROOT, download=True, min_rating=2.0)
    assert 1 not in train_data_loader_min_two[0].unique()
    assert 1 not in train_data_loader_min_two[120].unique()
    assert 3 in train_data_loader_min_two[0].unique()


def teardown_module():
    """
    Remove tempoary files after test execution
    """
    data_path = MOVIELENS_ROOT
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))
