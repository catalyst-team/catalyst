from pathlib import Path
import shutil

from git import Repo as repo

from catalyst import utils

URLS = {
    "classification": "https://github.com/catalyst-team/classification/",
    "segmentation": "https://github.com/catalyst-team/segmentation/",
    "detection": "https://github.com/catalyst-team/detection/"
}

CATALYST_ROOT = Path(__file__).resolve().parents[3]
PATH_TO_TEMPLATE = CATALYST_ROOT / "examples" / "_empty"


def clone_pipeline(
    template: str,
    out_dir: Path,
) -> None:
    """
    Clones pipeline from empty pipeline template or from demo pipelines
    available in Git repos of Catalyst Team.

    Args:
        template (str): type of pipeline you want to clone.
            empty/classification/segmentation
        out_dir (pathlib.Path): path where pipeline directory should be cloned
    Returns:
        None
    """
    if template == "empty" or template is None:
        utils.copy_directory(PATH_TO_TEMPLATE, out_dir)
    else:
        url = URLS[template]
        repo.clone_from(url, out_dir / "__git_temp")
        shutil.rmtree(out_dir / "__git_temp" / ".git")
        if (out_dir / "__git_temp" / ".gitignore").exists():
            (out_dir / "__git_temp" / ".gitignore").unlink()

        utils.copy_directory(out_dir / "__git_temp", out_dir)
        shutil.rmtree(out_dir / "__git_temp")
