from pathlib import Path
import shutil

from git import Repo as repo  # noqa: N813

from catalyst.utils.misc import copy_directory

URLS = {  # noqa: WPS407
    "classification": "https://github.com/catalyst-team/classification/",
    "segmentation": "https://github.com/catalyst-team/segmentation/",
    "detection": "https://github.com/catalyst-team/detection/",
}

CATALYST_ROOT = Path(__file__).resolve().parents[3]
PATH_TO_TEMPLATE = CATALYST_ROOT / "examples" / "_empty"


def clone_pipeline(template: str, out_dir: Path) -> None:
    """Clones pipeline from empty pipeline template or from demo pipelines
    available in Git repos of Catalyst Team.

    Args:
        template: type of pipeline you want to clone.
            empty/classification/segmentation
        out_dir: path where pipeline directory should be cloned
    """
    if template == "empty" or template is None:
        copy_directory(PATH_TO_TEMPLATE, out_dir)
    else:
        url = URLS[template]
        repo.clone_from(url, out_dir / "__git_temp")
        shutil.rmtree(out_dir / "__git_temp" / ".git")
        if (out_dir / "__git_temp" / ".gitignore").exists():
            (out_dir / "__git_temp" / ".gitignore").unlink()

        copy_directory(out_dir / "__git_temp", out_dir)
        shutil.rmtree(out_dir / "__git_temp")


__all__ = ["clone_pipeline"]
