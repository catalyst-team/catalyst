import shutil
from pathlib import Path
from ..tag2label import prepare_df_from_dirs


def prepare_dataset():
    shutil.rmtree('datasets', ignore_errors=True)

    # dummy datasets e.g. root1 and root2
    root1 = Path('datasets/root1')
    root1.mkdir(parents=True, exist_ok=True)
    root2 = Path('datasets/root2')
    root2.mkdir(parents=True, exist_ok=True)

    # dummy labels folders for root1
    root1_act1 = root1 / 'act1'
    root1_act2 = root1 / 'act2'
    root1_act1.mkdir()
    root1_act2.mkdir()

    # dummy labels folders for root2
    root2_act1 = root2 / 'act1'
    root2_act2 = root2 / 'act2'
    root2_act1.mkdir()
    root2_act2.mkdir()

    # dummy files for root1
    a = root1_act1 / 'a.txt'
    b = root1_act1 / 'b.txt'
    c = root1_act2 / 'c.txt'

    # dummy files for root2
    d = root2_act1 / 'd.txt'
    e = root2_act2 / 'e.txt'
    f = root2_act2 / 'f.txt'

    for file in [a, b, c, d, e, f]:
        file.touch()


def test_prepare_df_from_dirs_one():
    def check_filepath(f):
        return f.startswith('act1') or f.startswith('act2')

    prepare_dataset()
    df = prepare_df_from_dirs('datasets/root1', 'label')

    assert df.shape[0] == 3
    assert df.filepath.apply(check_filepath).sum().all()
    assert df.label.isin(['act1', 'act2']).all()

    shutil.rmtree('datasets', ignore_errors=True)


def test_prepare_df_from_dirs_multi():
    def check_filepath(f):
        return f.startswith('root1/act1') or \
               f.startswith('root1/act2') or \
               f.startswith('root2/act1') or \
               f.startswith('root2/act2')

    prepare_dataset()
    df = prepare_df_from_dirs(
        'datasets/root1,datasets/root2',
        'label')
    assert df.shape[0] == 6
    assert df.filepath.apply(check_filepath).sum().all()
    assert df.label.isin(['act1', 'act2']).all()

    shutil.rmtree('datasets', ignore_errors=True)
