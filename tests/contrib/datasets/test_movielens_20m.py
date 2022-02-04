import os
import shutil

import pytest

from catalyst.settings import SETTINGS
from tests import MOVIELENS20M_ROOT

if SETTINGS.ml_required and SETTINGS.is_torch_1_7_0:
    from catalyst.contrib.datasets import MovieLens20M

minversion = pytest.mark.skipif(
    not (SETTINGS.is_torch_1_7_0), reason="No catalyst[ml] required or torch version "
)


def setup_module():
    """
    Remove the temp folder if exists
    """
    data_path = MOVIELENS20M_ROOT
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))


@minversion
@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_download_split_by_user():
    """
    Test movielense download
    """
    MovieLens20M(MOVIELENS20M_ROOT, download=True, sample=True)

    filename = "ml-20m"

    # check if data folder exists
    assert os.path.isdir(MOVIELENS20M_ROOT) is True

    # cehck if class folder exists
    assert os.path.isdir(f"{MOVIELENS20M_ROOT}/MovieLens20M") is True

    # check if raw folder exists
    assert os.path.isdir(f"{MOVIELENS20M_ROOT}/MovieLens20M/raw") is True

    # check if processed folder exists
    assert os.path.isdir(f"{MOVIELENS20M_ROOT}/MovieLens20M/processed") is True

    # check some random file from MovieLens
    assert (
        os.path.isfile(
            f"{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv"
        )
        is True
    )

    # check if data file is not Nulll
    assert (
        os.path.getsize(
            f"{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv"
        )
        > 0
    )


@minversion
@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_download_split_by_ts():
    """
    Test movielense download
    """
    MovieLens20M(MOVIELENS20M_ROOT, download=True, split="ts", sample=True)

    filename = "ml-20m"

    # check if data folder exists
    assert os.path.isdir(MOVIELENS20M_ROOT) is True

    # cehck if class folder exists
    assert os.path.isdir(f"{MOVIELENS20M_ROOT}/MovieLens20M") is True

    # check if raw folder exists
    assert os.path.isdir(f"{MOVIELENS20M_ROOT}/MovieLens20M/raw") is True

    # check if processed folder exists
    assert os.path.isdir(f"{MOVIELENS20M_ROOT}/MovieLens20M/processed") is True

    # check some random file from MovieLens
    assert (
        os.path.isfile(
            f"{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv"
        )
        is True
    )

    # check if data file is not Nulll
    assert (
        os.path.getsize(
            f"{MOVIELENS20M_ROOT}/MovieLens20M/raw/{filename}/genome-scores.csv"
        )
        > 0
    )


@minversion
@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_minimal_ranking():
    """
    Tets retrieveing the minimal ranking
    """
    movielens_20m_min_two = MovieLens20M(
        MOVIELENS20M_ROOT, download=True, min_rating=2.0, sample=True, n_rows=1000000
    )

    assert 1 not in movielens_20m_min_two[1]._values().unique()
    assert 1 not in movielens_20m_min_two[3]._values().unique()
    assert (
        (2 in movielens_20m_min_two[1]._values().unique())
        or 3 in movielens_20m_min_two[1]._values().unique()
        or (4 in movielens_20m_min_two[1]._values().unique())
        or (5 in movielens_20m_min_two[1]._values().unique())
        or (len(movielens_20m_min_two[1]._values().unique()) == 0)
    )
    assert (
        (2 in movielens_20m_min_two[7]._values().unique())
        or (3 in movielens_20m_min_two[1]._values().unique())
        or (4 in movielens_20m_min_two[7]._values().unique())
        or (5 in movielens_20m_min_two[7]._values().unique())
        or (len(movielens_20m_min_two[1]._values().unique()) == 0)
    )
    assert (
        (3 in movielens_20m_min_two[3]._values().unique())
        or (4 in movielens_20m_min_two[3]._values().unique())
        or (5 in movielens_20m_min_two[3]._values().unique())
        or (len(movielens_20m_min_two[1]._values().unique()) == 0)
    )


@minversion
@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_users_per_item_filtering():
    """
    Tets retrieveing the minimal ranking
    """
    min_users_per_item = 2.0

    movielens_20m_min_users = MovieLens20M(
        MOVIELENS20M_ROOT,
        download=True,
        min_users_per_item=min_users_per_item,
        sample=True,
        n_rows=1000000,
    )

    assert (
        movielens_20m_min_users.users_activity["user_cnt"] >= min_users_per_item
    ).any()


@minversion
@pytest.mark.skipif(not (SETTINGS.ml_required), reason="No catalyst[ml] required")
def test_items_per_user_filtering():
    """
    Tets retrieveing the minimal ranking
    """
    min_items_per_user = 2.0
    min_users_per_item = 1.0
    movielens_20m_min_users = MovieLens20M(
        MOVIELENS20M_ROOT,
        download=True,
        min_items_per_user=min_items_per_user,
        min_users_per_item=min_users_per_item,
        sample=True,
        n_rows=1000000,
    )

    assert (
        movielens_20m_min_users.items_activity["item_cnt"] >= min_items_per_user
    ).any()


def teardown_module():
    """
    Remove tempoary files after test execution
    """
    data_path = MOVIELENS20M_ROOT
    try:
        shutil.rmtree(data_path)
    except Exception as e:
        print("Error! Code: {c}, Message, {m}".format(c=type(e).__name__, m=str(e)))
