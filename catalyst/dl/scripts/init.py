import argparse
import shutil
from git import Repo as repo
from pathlib import Path

from catalyst import utils


class CatalystInitException(Exception):
    def __init__(self, message):
        super().__init__(message)


def build_args(parser):
    parser.add_argument(
        "-p", "--pipeline",
        type=str, default=None,
        choices=["empty", "classification", "segmentation", "detection"],
        help="select a Catalyst pipeline"
    )
    parser.add_argument(
        "-o", "--out-dir",
        type=Path, default="./",
        help="path where to init"
    )

    return parser


URL = {
    "classification": "https://github.com/catalyst-team/classification/",
    "segmentation": "https://github.com/catalyst-team/segmentation/",
    "detection": "https://github.com/catalyst-team/detection/"
}

CATALYST_ROOT = Path(__file__).resolve().parents[3]
PATH_TO_TEMPLATE = CATALYST_ROOT / "examples" / "_empty"


def load_pipeline(
    url: str,
    out_dir: Path,
) -> None:
    repo.clone_from(url, out_dir / "__git_temp")
    shutil.rmtree(out_dir / "__git_temp" / ".git")
    if (out_dir / "__git_temp" / ".gitignore").exists():
        (out_dir / "__git_temp" / ".gitignore").unlink()

    utils.copy_directory(out_dir / "__git_temp", out_dir)
    shutil.rmtree(out_dir / "__git_temp")


def load_empty(out_dir: Path) -> None:
    utils.copy_directory(PATH_TO_TEMPLATE, out_dir)


def main(args, _):
    pipeline = args.pipeline
    if (pipeline is None) or (pipeline == "empty"):
        load_empty(args.out_dir)
    else:
        url = URL[pipeline]
        load_pipeline(url, args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, None)
