#!/usr/bin/env python
# flake8: noqa
# -*- coding: utf-8 -*-

# Note: To use the "upload" functionality of this file, you must:
#   $ pip install twine

import io
import os
from shutil import rmtree
import sys

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "catalyst"
DESCRIPTION = "Catalyst. PyTorch framework for DL research and development."
URL = "https://github.com/catalyst-team/catalyst"
EMAIL = "scitator@gmail.com"
AUTHOR = "Sergey Kolesnikov"
REQUIRES_PYTHON = ">=3.6.0"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    """Docs? Contribution is welcome."""
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


def load_readme():
    """Docs? Contribution is welcome."""
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with io.open(readme_path, encoding="utf-8") as f:
        return f"\n{f.read()}"


def load_version():
    """Docs? Contribution is welcome."""
    context = {}
    with open(os.path.join(PROJECT_ROOT, "catalyst", "__version__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        """Docs? Contribution is welcome."""
        pass

    def finalize_options(self):
        """Docs? Contribution is welcome."""
        pass

    def run(self):
        """Docs? Contribution is welcome."""
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(PROJECT_ROOT, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(load_version()))
        os.system("git push --tags")

        sys.exit()


# Specific dependencies.
extras = {
    "dev": load_requirements("requirements/requirements-dev.txt"),
    "cv": load_requirements("requirements/requirements-cv.txt"),
    "ml": load_requirements("requirements/requirements-ml.txt"),
    "nifti": load_requirements("requirements/requirements-nifti.txt"),
    "hydra": load_requirements("requirements/requirements-hydra.txt"),
    "optuna": load_requirements("requirements/requirements-optuna.txt"),
    "onnx": load_requirements("requirements/requirements-onnx.txt"),
    "onnx-gpu": load_requirements("requirements/requirements-onnx-gpu.txt"),
    "mlflow": load_requirements("requirements/requirements-mlflow.txt"),
    "neptune": load_requirements("requirements/requirements-neptune.txt"),
    "wandb": load_requirements("requirements/requirements-wandb.txt"),
    "fairscale": load_requirements("requirements/requirements-fairscale.txt"),
    "deepspeed": load_requirements("requirements/requirements-deepspeed.txt"),
    "albu": load_requirements("requirements/requirements-albu.txt"),
}
extras["all"] = extras["cv"] + extras["ml"] + extras["hydra"] + extras["optuna"]
# Meta dependency groups.
# all_deps = []
# for group_name in extras:
#     all_deps += extras[group_name]
# extras["all"] = all_deps

setup(
    name=NAME,
    version=load_version(),
    description=DESCRIPTION,
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    keywords=[
        "Machine Learning",
        "Distributed Computing",
        "Deep Learning",
        "Reinforcement Learning",
        "Computer Vision",
        "Natural Language Processing",
        "Recommendation Systems",
        "Information Retrieval",
        "PyTorch",
    ],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    download_url=URL,
    project_urls={
        "Bug Tracker": "https://github.com/catalyst-team/catalyst/issues",
        "Documentation": "https://catalyst-team.github.io/catalyst",
        "Source Code": "https://github.com/catalyst-team/catalyst",
    },
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "catalyst-contrib=catalyst.contrib.__main__:main",
            "catalyst-dl=catalyst.dl.__main__:main",
        ],
    },
    scripts=[
        "bin/scripts/catalyst-parallel-run",
        "bin/scripts/download-gdrive",
        "bin/scripts/extract-archive",
        "bin/scripts/install-apex",
    ],
    install_requires=load_requirements("requirements/requirements.txt"),
    extras_require=extras,
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        # Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Programming
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
