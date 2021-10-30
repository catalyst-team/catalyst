#!/usr/bin/env python
# flake8: noqa
# -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "catalyst"
DESCRIPTION = "Catalyst. Accelerated deep learning R&D with PyTorch."
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
    with open(readme_path, encoding="utf-8") as f:
        return f"\n{f.read()}"


def load_version():
    """Docs? Contribution is welcome."""
    context = {}
    with open(os.path.join(PROJECT_ROOT, "catalyst", "__version__.py")) as f:
        exec(f.read(), context)
    return context["__version__"]


# Specific dependencies.
extras = {
    "albu": load_requirements("requirements/requirements-albu.txt"),
    "cv": load_requirements("requirements/requirements-cv.txt"),
    "deepspeed": load_requirements("requirements/requirements-deepspeed.txt"),
    "dev": load_requirements("requirements/requirements-dev.txt"),
    "fairscale": load_requirements("requirements/requirements-fairscale.txt"),
    "hydra": load_requirements("requirements/requirements-hydra.txt"),
    "ml": load_requirements("requirements/requirements-ml.txt"),
    "mlflow": load_requirements("requirements/requirements-mlflow.txt"),
    "neptune": load_requirements("requirements/requirements-neptune.txt"),
    "nifti": load_requirements("requirements/requirements-nifti.txt"),
    "onnx-gpu": load_requirements("requirements/requirements-onnx-gpu.txt"),
    "onnx": load_requirements("requirements/requirements-onnx.txt"),
    "optuna": load_requirements("requirements/requirements-optuna.txt"),
    "wandb": load_requirements("requirements/requirements-wandb.txt"),
    # "xla": load_requirements("requirements/requirements-xla.txt"),
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
)
