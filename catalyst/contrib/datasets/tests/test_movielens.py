import os
import shutil

import pytest
import torch

from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    from catalyst.contrib.datasets import MovieLens


def setup_module():
    """
    Remove the temp folder if exists
    """
    data_path = "./data"
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))


@pytest.mark.skipif(
    not (SETTINGS.ml_required), reason="No catalyst[ml] required",
)
def test_download():
    """
    Test movielense download
    """
    MovieLens("./data", download=True)

    filename = "ml-100k"

    # check if data folder exists
    assert os.path.isdir("./data") is True

    # cehck if class folder exists
    assert os.path.isdir("./data/MovieLens") is True

    # check if raw folder exists
    assert os.path.isdir("./data/MovieLens/raw") is True

    # check if processed folder exists
    assert os.path.isdir("./data/MovieLens/processed") is True

    # check if raw folder exists
    assert os.path.isdir("./data/MovieLens/raw") is True

    # check some random file from MovieLens
    assert os.path.isfile("./data/MovieLens/raw/{}/u.info".format(filename)) is True

    # check Readme file from MovieLens
    assert os.path.isfile("./data/MovieLens/raw/{}/README".format(filename)) is True

    # check if data file is not Nulll
    assert os.path.getsize("./data/MovieLens/raw/{}/u.data".format(filename)) > 0

    # check if info file is not Nulll
    assert os.path.getsize("./data/MovieLens/raw/{}/u.info".format(filename)) > 0


@pytest.mark.skipif(
    not (SETTINGS.ml_required), reason="No catalyst[ml] required",
)
def test_reading():
    """
    Test reading of the movielens dataset
    """
    training_file = "training.pt"
    test_file = "test.pt"
    processed_folder = "data/MovieLens/processed"

    assert torch.load(os.path.join(processed_folder, test_file)).size() == torch.Size([943, 1682])
    assert torch.load(os.path.join(processed_folder, training_file)).size() == torch.Size(
        [943, 1682]
    )


@pytest.mark.skipif(
    not (SETTINGS.ml_required), reason="No catalyst[ml] required",
)
def test_minimal_ranking():
    """
    Tets retrieveing the minimal ranking
    """
    train_data_laoder_min_two = MovieLens("./data", min_rating=2.0)
    assert 1 not in train_data_laoder_min_two[0].unique()
    assert 1 not in train_data_laoder_min_two[120].unique()
    assert 3 in train_data_laoder_min_two[0].unique()


def teardown_module():
    """
    Remove tempoary files after test execution
    """
    data_path = "./data"
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))
