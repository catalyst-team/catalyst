import os
import shutil

import pytest

import torch

from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    from catalyst.contrib.datasets import MovieLens20M


def setup_module():
    """
    Remove the temp folder if exists
    """
    data_path = "./data"
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_download_split_by_user():
    """
    Test movielense download
    """
    MovieLens20M("./tmp_data", download=True, sample=True)

    filename = "ml-20m"

    # check if data folder exists
    assert os.path.isdir("./tmp_data") is True

    # cehck if class folder exists
    assert os.path.isdir("./tmp_data/MovieLens20M") is True

    # check if raw folder exists
    assert os.path.isdir("./tmp_data/MovieLens20M/raw") is True

    # check if processed folder exists
    assert os.path.isdir("./tmp_data/MovieLens20M/processed") is True

    # check some random file from MovieLens
    assert os.path.isfile("./tmp_data/MovieLens20M/raw/{}/genome-scores.csv".format(filename)) is True

    # check if data file is not Nulll
    assert os.path.getsize("./tmp_data/MovieLens20M/raw/{}/genome-scores.csv".format(filename)) > 0


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_download_split_by_ts():
    """
    Test movielense download
    """
    MovieLens20M("./tmp_data", download=True, split="ts", sample=True)

    filename = "ml-20m"

    # check if data folder exists
    assert os.path.isdir("./tmp_data") is True

    # cehck if class folder exists
    assert os.path.isdir("./tmp_data/MovieLens20M") is True

    # check if raw folder exists
    assert os.path.isdir("./tmp_data/MovieLens20M/raw") is True

    # check if processed folder exists
    assert os.path.isdir("./tmp_data/MovieLens20M/processed") is True

    # check some random file from MovieLens
    assert os.path.isfile("./tmp_data/MovieLens20M/raw/{}/genome-scores.csv".format(filename)) is True

    # check if data file is not Nulll
    assert os.path.getsize("./tmp_data/MovieLens20M/raw/{}/genome-scores.csv".format(filename)) > 0


@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_minimal_ranking():
    """
    Tets retrieveing the minimal ranking
    """
    movielens_20m_min_two = MovieLens20M("./tmp_data", min_rating=2.0, sample=True)

    assert 1 not in movielens_20m_min_two[1]._values().unique()
    assert 1 not in movielens_20m_min_two[3]._values().unique()
    assert 2 not in movielens_20m_min_two[4]._values().unique()
    assert 2 not in movielens_20m_min_two[7]._values().unique()
    assert ((3 in movielens_20m_min_two[1]._values().unique()) or (4 in movielens_20m_min_two[1]._values().unique()) or (5 in movielens_20m_min_two[1]._values().unique()))
    assert ((3 in movielens_20m_min_two[7]._values().unique()) or (4 in movielens_20m_min_two[7]._values().unique()) or (5 in movielens_20m_min_two[7]._values().unique()))
    assert ((3 in movielens_20m_min_two[3]._values().unique()) or (4 in movielens_20m_min_two[3]._values().unique()) or (5 in movielens_20m_min_two[3]._values().unique()))


def teardown_module():
    """
    Remove tempoary files after test execution
    """
    data_path = "./tmp_data"
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))
