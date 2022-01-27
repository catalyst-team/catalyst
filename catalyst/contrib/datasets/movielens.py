import itertools
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
from torch.utils.data import Dataset

from catalyst.contrib.datasets.misc import download_and_extract_archive


class MovieLens(Dataset):
    """
    MovieLens data sets were collected by the GroupLens Research Project
    at the University of Minnesota.

    This data set consists of:
    * 100,000 ratings (1-5) from 943 users on 1682 movies.
    * Each user has rated at least 20 movies.
    * Simple demographic info for the users
    (age, gender, occupation, zip)

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

    .. note::
        catalyst[ml] required for this dataset.
    """

    resources = (
        "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "0e33842e24a9c977be4e0107933c0723",
    )
    filename = "ml-100k.zip"
    training_file = "training.pt"
    test_file = "test.pt"

    def __init__(self, root, train=True, download=False, min_rating=0.0):
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

        Raises:
            RuntimeError: If ``download is False`` and the dataset not found.
        """
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)

        self.root = root
        self.train = train
        self.min_rating = min_rating

        if download:
            self._download()

        self._fetch_movies()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Set `download=True`")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, user_index):
        """Get item.

        Args:
            user_index (int): User index [0, 942]

        Returns:
            tensor: (items) item's ranking for the user with index user_index
        """
        return self.data[user_index]

    def __len__(self):
        """The length of the loader"""
        return self.dimensions[0]

    @property
    def raw_folder(self):
        """Create raw folder for data download

        Returns:
            raw_path (path): raw folder path
        """
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        """Create the folder for the processed files

        Returns:
            raw_path (path): processed folder path
        """
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def _check_exists(self):
        """Check if the path for tarining and testing data
        exists in processed folder.

        Returns:
            raw_path (path): processed folder path
        """
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def _download(self):
        """Download and extract files/"""
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

    def _read_raw_movielens_data(self):
        """Return the raw lines of the train and test files."""
        path = self.raw_folder

        with open(path + "/ml-100k/ua.base") as datafile:
            ua_base = datafile.read().split("\n")

        with open(path + "/ml-100k/ua.test") as datafile:
            ua_test = datafile.read().split("\n")

        with open(path + "/ml-100k/u.item", encoding="ISO-8859-1") as datafile:
            u_item = datafile.read().split("\n")

        with open(path + "/ml-100k/u.genre") as datafile:
            u_genre = datafile.read().split("\n")

        return (ua_base, ua_test, u_item, u_genre)

    def _build_interaction_matrix(self, rows, cols, data):
        """Builds interaction matrix.

        Args:
            rows (int): rows of the oevrall dataset
            cols (int): columns of the overall dataset
            data (generator object): generator of
            the data object

        Returns:
            interaction_matrix (torch.sparse.Float):
            sparse user2item interaction matrix
        """
        mat = sp.lil_matrix((rows, cols), dtype=np.int32)

        for uid, iid, rating, _ in data:
            if rating >= self.min_rating:
                mat[uid, iid] = rating
        coo = mat.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        interaction_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
        return interaction_matrix

    def _parse(self, data):
        """Parses the raw data. Substract one to shift to zero based indexing

        Args:
            data: raw data of the dataset

        Yields:
            Generator iterator for parsed data
        """
        for line in data:

            if not line:
                continue

            uid, iid, rating, timestamp = [int(x) for x in line.split("\t")]
            yield uid - 1, iid - 1, rating, timestamp

    def _get_dimensions(self, train_data, test_data):
        """Gets the dimensions of the raw dataset

        Args:
            train_data: (uid, iid, rating, timestamp)
                Genrator for training data
            test_data: (uid, iid, rating, timestamp)
                Genrator for testing data

        Returns:
            The total dimension of the dataset
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

    def _fetch_movies(self):
        """
        Fetch data and save in the pytorch format
            1. Read the train/test data from raw archive
            2. Parse train data
            3. Parse test data
            4. Save in the .pt with torch.save
        """
        data = self._read_raw_movielens_data()
        train_raw = data[0]
        test_raw = data[1]

        train_parsed = self._parse(train_raw)
        test_parsed = self._parse(test_raw)

        num_users, num_items = self._get_dimensions(train_parsed, test_parsed)

        train = self._build_interaction_matrix(
            num_users, num_items, self._parse(train_raw)
        )
        test = self._build_interaction_matrix(
            num_users, num_items, self._parse(test_raw)
        )
        assert train.shape == test.shape

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(train, f)

        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test, f)


class MovieLens20M(Dataset):
    """
    MovieLens data sets (ml-20m) were collected by
    the GroupLens Research Project at the University of Minnesota.

    This data set consists of:
    * 20,000,263 ratings (1-5)
    and 465,564 tag applications from 138,493 users on 27,278 movies.
    * Each user has rated at least 20 movies.
    * Simple demographic info for the users
    (age, gender, occupation, zip)

    Users were selected at random for inclusion.
    All selected users had rated at least 20 movies.
    No demographic information is included.
    Each user is represented by an id, and no other information is provided.

    More details about the contents and use of all these files follows.
    This and other GroupLens data sets are publicly available for download
    at http://grouplens.org/datasets/.

    The data was collected through the MovieLens web site.
    (movielens.umn.edu)  between January 09, 1995 and March 31, 2015.
    This dataset was generated on October 17, 2016.

    Neither the University of Minnesota nor any of the researchers involved
    can guarantee the correctness of the data, its suitability
    for any particular purpose, or the validity of
    results based on the use of the data set.

    The data set may be used for any research purposes
    under the following conditions:

    The user may not state or imply any endorsement
    from the University of Minnesota or the GroupLens Research Group.

    The user must acknowledge the use of the data set in
    publications resulting from the use of the data set
    (see below for citation information).

    The user may not redistribute the data without separate permission.

    The user may not use this information for any
    commercial or revenue-bearing purposes
    without first obtaining permission from a faculty member
    of the GroupLens Research Project at the University of Minnesota.

    The executable software scripts are provided "as is"
    without warranty of any kind, either expressed or implied, including,
    but not limited to, the implied warranties of merchantability
    and fitness for a particular purpose.

    The entire risk as to the quality and performance of them is with you.
    Should the program prove defective,
    you assume the cost of all necessary servicing, repair or correction.

    In no event shall the University of Minnesota,
    its affiliates or employees be liable to you for any damages
    arising out of the use or inability to use these programs (including
    but not limited to loss of data or data being rendered inaccurate).

    The data are contained in six files:
    1. genome-scores.csv
    2. genome-tags.csv
    3. links.csv
    4. movies.csv
    5. ratings.csv
    6. tags.csv

    Ratings Data File Structure (ratings.csv)
    All ratings are contained in the file ratings.csv.
    Each line of this file after the header row represents
    one rating of one movie by one user,and has the following format:

    1. userId,
    2. movieId,
    3. rating,
    4. timestamp

    Tags Data File Structure (tags.csv)

    1. userId,
    2. movieId,
    3. tag,
    4. timestamp

    Movies Data File Structure (movies.csv)

    1. movieId,
    2. title,
    3. genres

    Movie titles are entered manually or
    imported from https://www.themoviedb.org/, and include the year
    of release in parentheses.
    Errors and inconsistencies may exist in these titles.

    Links Data File Structure (links.csv)

    1. movieId,
    2. imdbId,
    3. tmdbId

    Tag Genome (genome-scores.csv and genome-tags.csv)

    1. movieId,
    2. tagId,
    3. relevance


    If you have any further questions or comments, please contact GroupLens
    <grouplens-info@cs.umn.edu>.
    https://files.grouplens.org/datasets/movielens/ml-20m-README.html
    """

    resources = (
        "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
        " cd245b17a1ae2cc31bb14903e1204af3",
    )
    filename = "ml-20m.zip"
    training_file = "training.pt"
    test_file = "test.pt"

    def __init__(
        self,
        root,
        train=True,
        download=False,
        min_rating=0.0,
        min_items_per_user=1.0,
        min_users_per_item=2.0,
        test_prop=0.2,
        split="users",
        sample=False,
        n_rows=1000,
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
            min_items_per_user (float, optional):
                Minimum number of items per user
                to include in the interaction matrix
            min_users_per_item (float, optional):
                Minimum rating to users per itemrs
                to include in the interaction matrix
            test_prop (float, optional): train-test split
            split (string, optional): the splittage method.
                `users` – split by users
                `ts` - split by timestamp
            sample (bool, optional):
                If true, then use the sample of the dataset.
                If true the `n_rows` shold be provide
            n_rows (int, optional): number of rows to retrieve.
                Availbale only with `sample = True`

        Raises:
            RuntimeError: If ``download = False`` and the dataset not found.
            RuntimeError: If torch version < `1.7.0`"
        """
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)

        self.root = root
        self.train = train
        self.min_rating = min_rating
        self.min_items_per_user = min_items_per_user
        self.min_users_per_item = min_users_per_item
        self.test_prop = test_prop
        self.nrows = n_rows
        self.sample = sample
        self.split = split

        if download:
            self._download()

        self._fetch_movies(split_by=split)

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Set `download=True`")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, user_index):
        """Get item.

        Args:
            user_index (int): User index

        Returns:
            tensor: (items) item's ranking for the user with index user_index
        """
        return self.data[user_index]

    def __len__(self):
        """The length of the loader"""
        return self.dimensions[0]

    @property
    def raw_folder(self):
        """Create raw folder for data download

        Returns:
            raw_path (path): raw folder path
        """
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self):
        """Create the folder for the processed files

        Returns:
            raw_path (path): processed folder path
        """
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def _check_exists(self):
        """Check if the path for tarining and testing data exists in
        processed folder.

        Returns:
            raw_path (path): processed folder path
        """
        return os.path.exists(
            os.path.join(self.processed_folder, self.training_file)
        ) and os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def _download(self):
        """Download and extract files"""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        url = self.resources[0]

        download_and_extract_archive(
            url=url,
            download_root=self.raw_folder,
            filename=self.filename,
            remove_finished=True,
        )

    def _read_raw_movielens_data(self):
        """Read the csv files with pandas.

        Returns:
            (movies, ratings, genome_scores, genome_tags, tags):
            (pd.DataFrame, pd.DataFrame, pd.DataFrame,
            pd.DataFrame, pd.DataFrame)
        """
        path = self.raw_folder

        if self.sample:
            movies = pd.read_csv(path + "/ml-20m/movies.csv", nrows=self.nrows)
            ratings = pd.read_csv(path + "/ml-20m/ratings.csv", nrows=self.nrows)
            genome_scores = pd.read_csv(
                path + "/ml-20m/genome-scores.csv", nrows=self.nrows
            )
            genome_tags = pd.read_csv(path + "/ml-20m/genome-tags.csv", nrows=self.nrows)
            tags = pd.read_csv(path + "/ml-20m/tags.csv", nrows=self.nrows)
        else:
            movies = pd.read_csv(path + "/ml-20m/movies.csv")
            ratings = pd.read_csv(path + "/ml-20m/ratings.csv")
            genome_scores = pd.read_csv(path + "/ml-20m/genome-scores.csv")
            genome_tags = pd.read_csv(path + "/ml-20m/genome-tags.csv")
            tags = pd.read_csv(path + "/ml-20m/tags.csv")

        return (movies, ratings, genome_scores, genome_tags, tags)

    def _build_interaction_matrix(self, ratings):
        """Builds interaction matrix.

        Args:
            ratings (pd.Dataframe): pandas DataFrame of the following format
                    userId	movieId	rating
                    20	1	924	     3.5
                    19	1	919	     3.5
                    86	1	2683	 3.5
                    61	1	1584	 3.5
                    23	1	1079	 4.0

        Returns:
            interaction_matrix (torch.sparse.Float):
            sparse user2item interaction matrix
        """
        csr_matrix = sp.coo_matrix(
            (
                ratings["rating"].astype(np.float32),
                (ratings["movieId"], ratings["userId"]),
            )
        )

        interaction_matrix = torch.sparse.LongTensor(
            torch.LongTensor([csr_matrix.row.tolist(), csr_matrix.col.tolist()]),
            torch.LongTensor(csr_matrix.data.astype(np.int32)),
        )

        return interaction_matrix

    def _parse(
        self,
        ratings,
        rating_cut=True,
        user_per_item_cut=True,
        item_per_user_cut=True,
        ts_cut=False,
    ):
        """Parses and pre-process the raw data.
        Substract one to shift to zero based indexing
        To-do add timestamp cut

        Args:
            ratings (pd.Dataframe): pandas DataFrame of the following format
                userId	movieId	rating	timestamp
                20	1	924	     3.5	1094785598
                19	1	919	     3.5	1094785621
                86	1	2683	 3.5	1094785650
                61	1	1584	 3.5	1094785656
                23	1	1079	 4.0	1094785665
            rating_cut (bool, optional):
                If true, filter datafreame on the `min_rating` value
            user_per_item_cut (bool, optional):
                If true, filter datafreame on the `min_users_per_item` value
            item_per_user_cut (bool, optional):
                If true, filter datafreame on the `min_items_per_user` value
            ts_cut (bool, optional):
                If true, filter datafreame on the `min_ts` value [TO-DO]

        Returns:
            ratings (pd.Dataframe): filtered `ratings` pandas DataFrame
            users_activity (pd.DataFrame):
                Number of items each user interacted with
            items_activity (pd.DataFrame):
                Number of users interacted with each item.
        """
        if rating_cut:
            ratings = ratings[ratings["rating"] >= self.min_rating].sort_values(
                ["userId", "timestamp"]
            )

        movie_id = "movieId"
        user_cnt_df = (
            ratings[[movie_id]]
            .groupby(movie_id, as_index=False)
            .size()
            .rename(columns={"size": "user_cnt"})
        )
        user_id = "userId"
        item_cnt_df = (
            ratings[[user_id]]
            .groupby(user_id, as_index=False)
            .size()
            .rename(columns={"size": "item_cnt"})
        )

        user_not_filtered = True
        item_not_filtered = True

        while user_not_filtered or item_not_filtered:
            ratings = ratings[
                ratings[movie_id].isin(
                    user_cnt_df.index[user_cnt_df["user_cnt"] >= self.min_users_per_item]
                )
            ]
            ratings = ratings[
                ratings[user_id].isin(
                    item_cnt_df.index[item_cnt_df["item_cnt"] >= self.min_items_per_user]
                )
            ]

            user_cnt_df = (
                ratings[[movie_id]]
                .groupby(movie_id, as_index=False)
                .size()
                .rename(columns={"size": "user_cnt"})
            )
            item_cnt_df = (
                ratings[[user_id]]
                .groupby(user_id, as_index=False)
                .size()
                .rename(columns={"size": "item_cnt"})
            )

            user_not_filtered = (user_cnt_df["user_cnt"] < self.min_users_per_item).any()
            item_not_filtered = (item_cnt_df["item_cnt"] < self.min_items_per_user).any()

        users_activity = (
            ratings[["userId"]]
            .groupby("userId", as_index=False)
            .size()
            .rename(columns={"size": "user_cnt"})
        )
        items_activity = (
            ratings[["movieId"]]
            .groupby("movieId", as_index=False)
            .size()
            .rename(columns={"size": "item_cnt"})
        )
        return ratings, users_activity, items_activity

    def _split_by_users(self, ratings, users_activity):
        """Split the ratings DataFrame into train and test
        Randomly shuffle users and split

        Args:
            ratings (pd.Dataframe): pandas DataFrame of the following format
                userId	movieId	rating	timestamp
                20	1	924	     3.5	1094785598
                19	1	919	     3.5	1094785621
                86	1	2683	 3.5	1094785650
                61	1	1584	 3.5	1094785656
                23	1	1079	 4.0	1094785665
            users_activity (pd.DataFrame):
                Number of items each user interacted with

        Returns:
            train_events (pd.Dataframe): pandas DataFrame for training data
            test_events (pd.Dataframe): pandas DataFrame for training data
        """
        idx_perm = np.random.permutation(users_activity.index.size)
        unique_uid = users_activity.index[idx_perm]
        n_users = unique_uid.size

        test_users = unique_uid[: int(n_users * self.test_prop)]
        train_users = unique_uid[int(n_users * self.test_prop) :]

        train_events = ratings.loc[ratings["userId"].isin(train_users)]
        test_events = ratings.loc[ratings["userId"].isin(test_users)]

        return (train_events, test_events)

    def _split_by_time(self, ratings):
        """Split the ratings DataFrame into train and test by timestamp
        Ratings[timestamp] extreme values used for the filtering interval

        Args:
            ratings (pd.Dataframe): pandas DataFrame of the following format
                userId	movieId	rating	timestamp
                20	1	924	     3.5	1094785598
                19	1	919	     3.5	1094785621
                86	1	2683	 3.5	1094785650
                61	1	1584	 3.5	1094785656
                23	1	1079	 4.0	1094785665

        Returns:
            train_events (pd.Dataframe): pandas DataFrame for training data
            test_events (pd.Dataframe): pandas DataFrame for training data
        """
        ts = ratings["timestamp"].sort_values()
        ts_max = ts.max()
        ts_min = ts.min()
        ts_split = ts_min + (ts_max - ts_min) * self.test_prop

        train_events = ratings[ratings["timestamp"] > ts_split]
        test_events = ratings[ratings["timestamp"] <= ts_split]

        return (train_events, test_events)

    def _fetch_movies(self, split_by="users"):
        """
        Fetch data and save in the pytorch format
            1. Read the MovieLens20 data from raw archive
            2. Parse the rating dataset
            3. Split dataset into train and test
            4. Build user-item matrix interaction
            5. Save in the .pt with torch.save

        Args:
            split_by (string, optional): the splittage method.
                `users` – split by users
                `ts` - split by timestamp

        Raises:
            ValueError: If `split_by` argument is not equal `users` or `ts`
        """
        raw_data = self._read_raw_movielens_data()

        ratings = raw_data[1]

        # TO-DO: add error handling
        ratings, users_activity, items_activity = self._parse(ratings)
        self.users_activity = users_activity
        self.items_activity = items_activity

        if split_by == "users":
            train_raw, test_raw = self._split_by_users(ratings, users_activity)
        if split_by == "ts":
            train_raw, test_raw = self._split_by_time(ratings)
        if split_by != "users" and split_by != "ts":
            raise ValueError("Only splitting by users and ts supported")

        train = self._build_interaction_matrix(train_raw)
        test = self._build_interaction_matrix(test_raw)

        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(train, f)

        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test, f)


__all__ = ["MovieLens", "MovieLens20M"]
