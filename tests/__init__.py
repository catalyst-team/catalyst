# flake8: noqa
# Needed to collect coverage data
import os
import path

DATA_ROOT = path.Path(os.path.abspath(os.path.dirname(__file__))).abspath()
MOVIELENS_ROOT = (
    path.Path(os.path.abspath(os.path.dirname(__file__)))
    .joinpath("./movielens")
    .abspath()
)
MOVIELENS20M_ROOT = (
    path.Path(os.path.abspath(os.path.dirname(__file__)))
    .joinpath("./movielens20m")
    .abspath()
)
