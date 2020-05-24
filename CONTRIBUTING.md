# Contribution guide

## How to strart?

Contributing is quite easy: suggest ideas and make them done.
We use [GitHub issues](https://github.com/catalyst-team/catalyst/issues) for bug reports and feature requests.

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
