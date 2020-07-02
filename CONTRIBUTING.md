# Contribution guide

## How to start?

Contributing is quite easy: suggest ideas and make them done.
We use [GitHub issues](https://github.com/catalyst-team/catalyst/issues) for bug reports and feature requests.

Every good PR is usually consists of:
- feature implementation :)
- documentation to describe this feature to other people
- tests to ensure everything is implemented correctly
- `CHANGELOG.md` update for framework development history

You can check these examples as a good example to follow:
- https://github.com/catalyst-team/catalyst/pull/855
- https://github.com/catalyst-team/catalyst/pull/858
- https://github.com/catalyst-team/catalyst/pull/842
- https://github.com/catalyst-team/catalyst/pull/825
- https://github.com/catalyst-team/catalyst/pull/862

#### Step-by-step guide

##### New feature

1. Make an issue with your feature description;
2. We shall discuss the design and its implementation details;
3. Once we agree that the plan looks good, go ahead and implement it.


##### Bugfix

1. Goto [GitHub issues](https://github.com/catalyst-team/catalyst/issues);
2. Pick an issue and comment on the task that you want to work on this feature;
3. If you need more context on a specific issue, please ask, and we will discuss the details.


You can also join our [Catalyst slack](https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw) to make it easier to discuss.
Once you finish implementing a feature or bugfix, please send a Pull Request.

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


##### Contribution best practices

1. Install requirements
    ```
    brew install bash # for MacOS users, as we need bash version >= 4.0.0
    pip install -r requirements/requirements.txt -r requirements/requirements-dev.txt
    # for easy-to-go development, we suggest to install extra dependencies
    pip install -r requirements/requirements-ml.txt -r requirements/requirements-cv.txt -r requirements/requirements-nlp.txt
    ```
2. Break your work into small, single-purpose updates if possible.
It's much harder to merge in a large change with a lot of disjoint features.
3. Submit the update as a GitHub pull request against the `master` branch.
4. Make sure that you provide docstrings for all your new methods and classes.
5. Add new unit tests for your code.
6. Check the [codestyle](#codestyle)
7. Make sure that your code [passes the unit tests](#unit-tests)


#### Codestyle

Do not forget to check the codestyle for your PR with

```bash
catalyst-make-codestyle && catalyst-check-codestyle
```

Make sure to have your python packages complied with `requirements/requirements.txt` and `requirements/requirements-dev.txt` to get codestyle run clean.

#### Unit tests

Do not forget to check that your code passes the unit tests

```bash
pytest .
```

##### Adding new tests

Create a new bash file in `bin/tests` with tests for your new feature.
If file name starts with `check_dl_core`, `check_dl_cv` or `check_dl_nlp` then your new tests will be executed
automaticaly on pull request, otherwise you need to update `bin/tests/check_dl_all.sh`.


##### Testing Notebook API

The easiest way to test Notebook API is to test expected behaviour directly in python.
It can be done in different ways and one of them is to execute python script with `-c`:

```bash
python -c "assert True != False"
```

If your feature affects output files - please check that directory with logs contains all required files.

##### Testing Config API

Create a folder with tests in `tests` directory and define there minimal required files - `__init__.py`,
`experiment.py`, `models.py` and config files (like `configN.yml`) with test configuration. Your folder name
should represent the part of API you are testing - for example, if I want to test dl part I will call a new
test folder like `_tests_dl_my_awesome_new_feature`.

As was mentioned previously, if your feature affects output files - please add tests for required files in
directory with logs.

If your feature affects some metrics - you need to check that everything works as expected during
epochs and/or stages. You can do this with `<logdir>/checkpoints/_metrics.json` file (load this file
in python and check values or something similar) or `<logdir>/log.txt`.


## Documentation

Catalyst uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting [docstrings](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

How to setup Google style documentation style in PyCharm:
[![Catalyst logo](https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/pycharm-google-style.png)](https://github.com/catalyst-team/catalyst)


#### Check that you have written working docs

Make the docs with
```bash
rm -rf ./builds; REMOVE_BUILDS=0 make check-docs
```

Now you can open them into your browser, for example with
```bash
vivaldi-stable ./builds/index.html
```

If you have some issues with building docs - please make sure that you installed required pip packages.

##### Check that you have written working docs with Docker

The command requires `Sphinx` and some sphinx-specific libraries.
If you don't want to install them, you may make a `catalyst-dev` container
```bash
make docker-dev
# and then run test
docker run \
    -v `pwd`/:/workspace/ \
    catalyst-dev:latest \
    bash -c "make check-docs"
```
