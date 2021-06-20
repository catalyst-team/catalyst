# flake8: noqa
# import pandas as pd
# import pytest
#
# from catalyst.contrib.utils.pandas import (
#     folds_to_list,
#     split_dataframe_on_stratified_folds,
# )
#
#
# def test_folds_to_list():
#     """@TODO: Docs. Contribution is welcome."""
#     assert folds_to_list("1,2,1,3,4,2,4,6") == [1, 2, 3, 4, 6]
#     assert folds_to_list([1, 2, 3.0, 5, 2, 1]) == [1, 2, 3, 5]
#     assert folds_to_list([]) == []
#
#     with pytest.raises(ValueError):
#         folds_to_list([1, "True", 3.0, None, 2, 1])
#
#
# def _setup_data(num_rows=10):
#     df_data = []
#     for i in range(num_rows):
#         if i < (num_rows / 2):
#             df_data.append(["ants", "%s.jpg" % i, 0])
#         else:
#             df_data.append(["bees", "%s.jpg" % i, 1])
#     return pd.DataFrame(df_data, columns=["tag", "filepath", "class"])
#
#
# def test_stratified_fold_split():
#     """@TODO: Docs. Contribution is welcome."""
#     df = _setup_data()
#
#     splitted = split_dataframe_on_stratified_folds(
#         dataframe=df, class_column="class"
#     )
#
#     assert int == splitted["fold"].dtype
#     assert set(range(5)) == set(splitted["fold"].unique())
#     ants_folds = set(splitted[splitted["tag"] == "ants"]["fold"])
#     bees_folds = set(splitted[splitted["tag"] == "bees"]["fold"])
#     assert ants_folds == bees_folds
#
#
# def test_stratified_fold_split_num_folds():
#     """@TODO: Docs. Contribution is welcome."""
#     df = _setup_data()
#
#     splitted = split_dataframe_on_stratified_folds(df, "class", n_folds=2)
#
#     assert set(range(2)) == set(splitted["fold"].unique())
