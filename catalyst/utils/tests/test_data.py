import pandas as pd
from catalyst.utils import data


def _setup_data(num_rows=10):
    df_data = []
    for i in range(num_rows):
        if i < (num_rows/2):
            df_data.append(['ants', '%s.jpg' % i, 0])
        else:
            df_data.append(['bees', '%s.jpg' % i, 1])
    return pd.DataFrame(df_data, columns=['tag', 'filepath', 'class'])


def test_stratified_fold_split():
    df = _setup_data()

    splitted = data.stratified_fold_split(dataframe=df, class_column='class')

    assert int == splitted['fold'].dtype
    assert set(range(5)) == set(splitted['fold'].unique())
    ants_folds = set(splitted[splitted['tag'] == 'ants']['fold'])
    bees_folds = set(splitted[splitted['tag'] == 'bees']['fold'])
    assert ants_folds == bees_folds


def test_stratified_fold_split_num_folds():
    df = _setup_data()

    splitted = data.stratified_fold_split(df, 'class', n_folds=2)

    assert set(range(2)) == set(splitted['fold'].unique())
