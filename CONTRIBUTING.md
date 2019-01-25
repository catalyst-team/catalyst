# Contribution guide

## Issues

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


Once you finish implementing a feature or bugfix, please send a Pull Request.

If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


##### Contribution best practices

1. Break your work into small, single-purpose updates if possible. 
It's much harder to merge in a large change with a lot of disjoint features.
2. Submit the update as a GitHub pull request against the `master` branch.
3. Make sure that you provide docstrings for all your new methods and classes
4. Make sure that your code passes the unit tests.
5. Add new unit tests for your code.

#### Codestyle

Do not forget to check the codestyle for your PR with

- flake
    ```bash
    flake8 . --count --ignore=E126,E226,E704,E731,W503,W504 --max-complexity=16 --show-source --statistics
    ```
- yapf
    ```bash
    ./yapf.sh --all-in-place
    ```

or with

```bash
./codestyle.sh
```

## Documentation

Catalyst uses [Google style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting docstrings. 
Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.
