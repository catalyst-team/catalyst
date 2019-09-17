import argparse
import os
import shutil
import subprocess
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

    utils.boolean_flag(
        parser, "force",
        default=False,
        help="force init"
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
    with open(os.devnull, "w") as devnull:
        try:
            subprocess.run(
                f"git clone {url} {out_dir / 'temp'}".split(),
                shell=True, stderr=devnull
            )

            with open(os.devnull, "w") as devnull:
                subprocess.run(
                    f"mv {out_dir / 'temp'} {out_dir}".split(),
                    shell=True, stderr=devnull
                )

        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            raise CatalystInitException(
                f"Error cloning from {url} to {out_dir}"
            ) from ex

    shutil.rmtree(out_dir / ".git")

    if (out_dir / ".gitignore").exists():
        (out_dir / ".gitignore").unlink()


def load_empty(out_dir: Path) -> None:
    shutil.copytree(PATH_TO_TEMPLATE, out_dir / "temp")
    with open(os.devnull, "w") as devnull:
        print(f"mv {out_dir / 'temp'}/* {out_dir}/")
        subprocess.run(
            f"mv {out_dir / 'temp'}/* {out_dir}/".split()
            # shell=True
        )

        subprocess.run(
            f"rm -rf {out_dir / 'temp'}".split()
            # shell=True
        )


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
