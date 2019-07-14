from ..tag2label import prepare_df_from_dirs


def _setup_dataset_fs(tmp_path):
    def create_children(parent, children):
        for child, sub_children in children.items():
            sub_parent = parent / child
            sub_parent.mkdir()
            if type(sub_children) == list:
                [
                    (sub_parent / sub_child).touch()
                    for sub_child in sub_children
                ]
            else:
                create_children(sub_parent, sub_children)

    fs_structure = {
        "datasets": {
            "root1": {
                "act1": ["a.txt", "b.txt"],
                "act2": ["c.txt"]
            },
            "root2": {
                "act1": ["d.txt"],
                "act2": ["e.txt", "f.txt"]
            },
        }
    }

    create_children(tmp_path, fs_structure)


def test_prepare_df_from_dirs_one(tmp_path):
    def check_filepath(filepath):
        return filepath.startswith("act1") or filepath.startswith("act2")

    _setup_dataset_fs(tmp_path)
    root_path = tmp_path / "datasets/root1"
    df = prepare_df_from_dirs(str(root_path), "label")

    assert df.shape[0] == 3
    assert df.filepath.apply(check_filepath).sum().all()
    assert df.label.isin(["act1", "act2"]).all()


def test_prepare_df_from_dirs_multi(tmp_path):
    def check_filepath(filepath):
        return filepath.startswith("root1/act1") or \
               filepath.startswith("root1/act2") or \
               filepath.startswith("root2/act1") or \
               filepath.startswith("root2/act2")

    _setup_dataset_fs(tmp_path)
    ds_path = tmp_path / "datasets"
    root_paths = ",".join([str(ds_path / "root1"), str(ds_path / "root2")])
    df = prepare_df_from_dirs(root_paths, "label")

    assert df.shape[0] == 6
    assert df.filepath.apply(check_filepath).sum().all()
    assert df.label.isin(["act1", "act2"]).all()
