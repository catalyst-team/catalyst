import os
import zipfile

import numpy as np

import torch
from torch.utils.data import Dataset

from catalyst.contrib.datasets.utils import download_and_extract_archive


class MovieLens(Dataset):
    '''
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
    '''

    resources = (
            "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
            "6f5ca7e518b6970ec2265ce66a80ffdc"
            )
    filename = 'ml-100k'


    def __init__(self):
        pass

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
    
    def _downlaod(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

    def _get_raw_movielens_data(self):
        """
            Download movielens data if it doesn't exsit
        """

        path = self._get_movielens_path()

        if not os.path.isfile(path):
            download_and_extract_archive(
                url = self.resources[0]
                download_root = 
            )

        with zipfile.ZipFile(path) as datafile:
            return (datafile.read('ml-100k/ua.base').decode().split('\n'),
                datafile.read('ml-100k/ua.test').decode().split('\n'))

        if self._check_exist():
            return
