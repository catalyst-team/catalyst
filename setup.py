#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the "upload" functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "catalyst"
DESCRIPTION = "Catalyst. High-level utils for PyTorch DL & RL research."
URL = "https://github.com/catalyst-team/catalyst"
EMAIL = "scitator@gmail.com"
AUTHOR = "Sergey Kolesnikov"
REQUIRES_PYTHON = ">=3.6.0"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()


def load_readme():
    readme_path = os.path.join(PROJECT_ROOT, "README.md")
    with io.open(readme_path, encoding="utf-8") as f:
        return "\n" + f.read()


def load_version():
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
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(PROJECT_ROOT, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system(
            "{0} setup.py sdist bdist_wheel --universal".format(sys.executable)
        )

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(load_version()))
        os.system("git push --tags")

        sys.exit()


# Specific dependencies.
extras = {
    "contrib": load_requirements("requirements/requirements-contrib.txt"),
    "rl": load_requirements("requirements/requirements-rl.txt"),
}


# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras["all"] = all_deps


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
        "PyTorch",
    ],
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "catalyst-dl=catalyst.dl.__main__:main",
            "catalyst-rl=catalyst.rl.__main__:main",
            "catalyst-contrib=catalyst.contrib.__main__:main",
            "catalyst-data=catalyst.data.__main__:main",
        ],
    },
    scripts=[
        "bin/catalyst-parallel-run",
        "bin/catalyst-rl-run",
    ],
    install_requires=load_requirements("requirements/requirements.txt"),
    extras_require=extras,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
