import torch
import os
import zipfile

from catalyst.contrib.datasets import MovieLens

def test_download():

    train_dataL_laoder = MovieLens(
                    "./data", 
                    download=True
                    )

    test_data_loader = MovieLens(
                    "./data",
                    train=False
                    )
    
    # check if data folder exists
    assert os.path.isdir("./data") == True

    # cehck if class folder exists
    assert os.path.isdir("./data/MovieLens") == True

    # check if raw folder exisst
    assert os.path.isdir("./data/MovieLens/raw") == True

    # check if processed folder exisst
    assert os.path.isdir("./data/MovieLens/processed") == True

    # check if raw folder exisst
    assert os.path.isdir("./data/MovieLens/raw") == True

    # check some random file from MovieLense
    assert os.path.isfile("./data/MovieLens/raw/u.info") == True

    # check Readme file from MovieLense
    assert os.path.isfile("./data/MovieLens/raw/README") == True

    #check if data file is not Nulll
    assert os.path.getsize("./data/MovieLens/raw/u.data") > 0

    #check if info file is not Nulll
    assert os.path.getsize("./data/MovieLens/raw/u.info") > 0