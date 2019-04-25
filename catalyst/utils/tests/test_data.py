import pandas as pd
from catalyst.utils import data


def test_stratified_fold_split():
    df_data = []
    for i in range(10):
        if i < 5:
            df_data.append(['ants', '%s.jpg' % i, 0])
        else:
            df_data.append(['bees', '%s.jpg' % i, 1])
    df = pd.DataFrame(df_data, columns=['tag', 'filepath', 'class'])

    splitted = data.stratified_fold_split(dataframe=df, class_column='class')

    assert splitted['fold'].dtype == int
