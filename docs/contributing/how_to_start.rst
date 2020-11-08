How to start
==============================================================================

Issues
------

We use `GitHub issues`_ for bug reports and feature requests.

Step-by-step guide
^^^^^^^^^^^^^^^^^^

New feature
'''''''''''

1. Make an issue with your feature description;
2. We shall discuss the design and its implementation details;
3. Once we agree that the plan looks good, go ahead and implement it.

Bugfix
''''''

1. Goto `GitHub issues`_;
2. Pick an issue and comment on the task that you want to work on this
   feature;
3. If you need more context on a specific issue, please ask, and we will
   discuss the details.

Once you finish implementing a feature or bugfix, please send a Pull
Request.

If you are not familiar with creating a Pull Request, here are some
guides:

- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/

Contribution best practices
'''''''''''''''''''''''''''

1. Install requirements

.. code-block:: bash

    brew install bash # for MacOS users
    pip install -r requirements/requirements.txt -r requirements/requirements-dev.txt

2. Break your work into small, single-purpose updates if possible. It's much harder to merge in a large change with a lot of disjoint features.
3. Submit the update as a GitHub pull request against the `master` branch.
4. Make sure that you provide docstrings for all your new methods and classes.
5. Add new unit tests for your code.
6. Check the codestyle
7. Make sure that your code passes the unit tests


Codestyle
^^^^^^^^^

Do not forget to check the codestyle for your PR with

.. code-block:: bash

    catalyst-make-codestyle && catalyst-check-codestyle

Make sure to have your python packages complied with `requirements/requirements.txt` and `requirements/requirements-dev.txt` to get codestyle run clean.


Unit tests
^^^^^^^^^^

Do not forget to check that your code passes the unit tests

.. code-block:: bash

    pytest .


Documentation
-------------

Catalyst uses `Google style`_ for formatting `docstrings`_. Length of line
inside docstrings block must be limited to 80 characters to fit into
Jupyter documentation popups.

Check that you have written working docs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    make check-docs

The command requires Sphinx and some sphinx-specific libraries.
If you don't want to install them, you may make a catalyst-dev container

.. code-block:: bash

    make docker-dev
    docker run \\
        -v `pwd`/:/workspace/ \\
        catalyst-dev:latest \\
        bash -c "make check-docs"

To build docs add environment variable REMOVE_BUILDS=0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    REMOVE_BUILDS=0 make check-docs

or through docker

.. code-block:: bash

    docker run \\
        -v `pwd`/:/workspace/ \\
        catalyst-dev:latest \\
        bash -c "REMOVE_BUILDS=0 make check-docs"

The docs will be stored in `builds/` folder.


.. _GitHub issues: https://github.com/catalyst-team/catalyst/issues
.. _Google style: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _docstrings: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings