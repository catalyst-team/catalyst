import os
import zipfile
import logging
import itertools

import numpy as np

import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

from catalyst.contrib.datasets.utils import download_and_extract_archive, download_url


class MovieLens(Dataset):
    """
        MovieLens data sets were collected by the GroupLens Research Project
        at the University of Minnesota.
        
        This data set consists of:
            * 100,000 ratings (1-5) from 943 users on 1682 movies. 
            * Each user has rated at least 20 movies. 
                * Simple demographic info for the users (age, gender, occupation, zip)

        The data was collected through the MovieLens web site
        (movielens.umn.edu) during the seven-month period from September 19th, 
        1997 through April 22nd, 1998. This data has been cleaned up - users
        who had less than 20 ratings or did not have complete demographic
        information were removed from this data set. Detailed descriptions of
        the data file can be found at the end of this file.

        Neither the University of Minnesota nor any of the researchers
        involved can guarantee the correctness of the data, its suitability
        for any particular purpose, or the validity of results based on the
        use of the data set.  The data set may be used for any research
        purposes under the following conditions:

            * The user may not state or imply any endorsement from the
            University of Minnesota or the GroupLens Research Group.

            * The user must acknowledge the use of the data set in
            publications resulting from the use of the data set
            (see below for citation information).

            * The user may not redistribute the data without separate
            permission.

            * The user may not use this information for any commercial or
            revenue-bearing purposes without first obtaining permission
            from a faculty member of the GroupLens Research Project at the
            University of Minnesota.

        If you have any further questions or comments, please contact GroupLens
        <grouplens-info@cs.umn.edu>. 
        http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
    """

    resources = (
        "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "0e33842e24a9c977be4e0107933c0723"
    )
    filename = "ml-100k.zip"
    training_file = "training.pt"
    test_file = "test.pt"

    def __init__(
        self, 
        root,
        train=True,
        download=False,
        min_rating = 0.0
    ):
        """
        Args:
            root (string): Root directory of dataset where
                ``MovieLens/processed/training.pt``
                and  ``MovieLens/processed/test.pt`` exist.
            train (bool, optional): If True, creates dataset from
                ``training.pt``, otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from
                the internet and puts it in root directory. If dataset
                is already downloaded, it is not downloaded again.
            min_rating (float, optional): Minimum rating to include in 
                the interaction matrix
        """

        if isinstance(root, torch._six.string_classes):  # noqa: WPS437
            root = os.path.expanduser(root)
        
        self.root = root
        self.train = train
        self.min_rating = min_rating

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )
        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        self.data = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    def __getitem__(self, item_index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """ 
        return self.data.tocsr()[item_index, item_index]

    def __len__(self):
        """@TODO: Docs. Contribution is welcome."""
        return self.dimensions[0]

    @property
    def raw_folder(self):
        """
            Create raw folder for data download
        """
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        """@TODO: Docs. Contribution is welcome."""
        return os.path.join(self.root, self.__class__.__name__, "processed")


    def _check_exists(self):

        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(
            os.path.join(self.processed_folder, self.test_file)
        )

    def _download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        url = self.resources[0]
        md5 = self.resources[1]

        download_and_extract_archive(
            url=url,
            download_root=self.raw_folder,
            filename=self.filename,
            md5=md5,
            remove_finished=True,
        )
        
        print("Processing")

        

    def _read_raw_movielens_data(self):
        """
        Return the raw lines of the train and test files
        """

        path = self.raw_folder

        with open(path+"/ml-100k/ua.base") as datafile:
            ua_base = datafile.read().split("\n")

        with open(path+"/ml-100k/ua.test") as datafile:
            ua_test = datafile.read().split("\n")
        
        with open(path+"/ml-100k/u.item", encoding = "ISO-8859-1") as datafile:
            u_item = datafile.read().split("\n")

        with open(path+"/ml-100k/u.genre") as datafile:
            u_genre = datafile.read().split("\n")

        return (
            ua_base,
            ua_test,
            u_item,
            u_genre
        )

    def _build_interaction_matrix(self, rows, cols, data):

        mat = sp.lil_matrix((rows, cols), dtype=np.int32)

        for uid, iid, rating, _ in data:
            if rating >= self.min_rating:
                mat[uid, iid] = rating

        return mat.tocoo()

    def _parse(self, data):
        """
        Parse the raw data. 
        Substract one to shift to zero based indexing
        """

        for line in data:

            if not line:
                continue

            uid, iid, rating, timestamp = [int(x) for x in line.split("\t")]
            yield uid - 1, iid - 1, rating, timestamp


    def _get_dimensions(self, train_data, test_data):
        """
        Get the dimensions of the raw dataset
        """

        uids = set()
        iids = set()

        for uid, iid, _, _ in itertools.chain(train_data, test_data):
            uids.add(uid)
            iids.add(iid)

        rows = max(uids) + 1
        cols = max(iids) + 1

        self.dimensions = (rows, cols)

        return rows, cols

    
    def _parse_item_metadata(self, num_items, item_metadata_raw, genres_raw):
        """
        Get item metadata
        """
        genres = []

        for line in genres_raw:
            if line:
                genre, gid = line.split("|")
                genres.append("genre:{}".format(genre))

        id_feature_labels = np.empty(num_items, dtype=np.object)
        genre_feature_labels = np.array(genres)

        id_features = sp.identity(num_items, format="csr", dtype=np.float32)
        genre_features = sp.lil_matrix((num_items, len(genres)), dtype=np.float32)

        for line in item_metadata_raw:

            if not line:
                continue

            splt = line.split("|")

            # Zero-based indexing
            iid = int(splt[0]) - 1
            title = splt[1]

            id_feature_labels[iid] = title

            item_genres = [idx for idx, val in enumerate(splt[5:]) if int(val) > 0]

            for gid in item_genres:
                genre_features[iid, gid] = 1.0

        return (
            id_features,
            id_feature_labels,
            genre_features.tocsr(),
            genre_feature_labels,
        )


    def fetch_movies(self):
        train_raw, test_raw, item_metadata_raw, genres_raw = self._read_raw_movielens_data()
        parsed_train = self._parse(train_raw)
        parsed_test = self._parse(test_raw)
        num_users, num_items = self._get_dimensions(parsed_train, parsed_test)
        train = self._build_interaction_matrix(num_users, num_items, parsed_train)
        test = self._build_interaction_matrix(num_users, num_items, parsed_test)
        assert train.shape == test.shape

        with open(
            os.path.join(self.processed_folder, self.training_file), "wb"
        ) as f:
            torch.save(train, f)

        with open(
            os.path.join(self.processed_folder, self.test_file), "wb"
        ) as f:
            torch.save(test, f)



        # with open(
        #     os.path.join(self.processed_folder, self.training_file), "wb"
        # ) as f:
        #     torch.save(parsed_train, f)
        
        # with open(
        #     os.path.join(self.processed_folder, self.training_file), "wb"
        # ) as f:
        #     torch.save(parsed_test, f)


def main():
    root = "./data"
    movielens = MovieLens(root, download=False)
    movielens.fetch_movies()
    print(len(movielens))
    print(movielens[4])


if  __name__ == "__main__":
    main()
        