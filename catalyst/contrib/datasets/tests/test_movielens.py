import torch
import os
import zipfile

from catalyst.contrib.datasets import MovieLens

def test_download():

    data_path = './data'
    try:
        os.rmdir(data_path)
    except OSError as e:
        print("Error: %s : %s" % (data_path, e.strerror))

    MovieLens(
            "./data", 
            download=True
            )

    filename = "ml-100k"
    
    # check if data folder exists
    assert os.path.isdir("./data") == True

    # cehck if class folder exists
    assert os.path.isdir("./data/MovieLens") == True

    # check if raw folder exists
    assert os.path.isdir("./data/MovieLens/raw") == True

    # check if processed folder exists
    assert os.path.isdir("./data/MovieLens/processed") == True

    # check if raw folder exists
    assert os.path.isdir("./data/MovieLens/raw") == True

    # check some random file from MovieLens
    assert os.path.isfile("./data/MovieLens/raw/{}/u.info".format(filename)) == True

    # check Readme file from MovieLens
    assert os.path.isfile("./data/MovieLens/raw/{}/README".format(filename)) == True

    #check if data file is not Nulll
    assert os.path.getsize("./data/MovieLens/raw/{}/u.data".format(filename)) > 0

    #check if info file is not Nulll
    assert os.path.getsize("./data/MovieLens/raw/{}/u.info".format(filename)) > 0


def test_reading():
    training_file = "training.pt"
    test_file = "test.pt"
    processed_folder = "data/MovieLens/processed"

    assert torch.load(os.path.join(processed_folder, test_file)).size() == torch.Size([943, 1682])
    assert torch.load(os.path.join(processed_folder, training_file)).size() == torch.Size([943, 1682])


def test_minimal_ranking():
    train_dataL_laader_no_min = MovieLens(
            "./data", 
            )
    train_dataL_laoder_min_two = MovieLens(
                "./data", 
                min_rating = 2.0
                )
    assert (1 not in train_dataL_laoder_min_two[0].unique())
    assert (1 not in train_dataL_laoder_min_two[120].unique())
    assert (3 in train_dataL_laoder_min_two[0].unique())



    