import argparse
from pathlib import Path
import shutil

from git import Repo as repo

from catalyst import utils
from catalyst.dl.utils import clone_pipeline, run_wizard


def build_args(parser):
    parser.add_argument(
        "-p", "--pipeline",
        type=str, default=None,
        choices=["empty", "classification", "segmentation", "detection"],
        help="select a Catalyst pipeline"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="use interactive wizard to setup Catalyst pipeline"
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
    if args.interactive:
        run_wizard()
    else:
        clone_pipeline(args.pipeline, args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args, None)
