Logger
==============================================================================

Speaking about the logging, Catalyst united the monitoring system API support into one abstraction:

.. image:: https://raw.githubusercontent.com/Scitator/catalyst21-post-pics/main/code_logger21.png
    :alt: Runner


With such a simple API,
we already provide integrations for `Tensorboard`_ and `MLFlow`_ monitoring systems.
More advanced loggers for Neptune and Wandb with artifacts and hyperparameters storing
are in development thanks to joint collaborations between our teams!
All currently supported loggers you could find under the `Logger API section`_.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`Tensorboard`: https://catalyst-team.github.io/catalyst/api/loggers.html#tensorboardlogger
.. _`MLFlow`: https://catalyst-team.github.io/catalyst/api/loggers.html#mlflowlogger
.. _`Logger API section`: https://catalyst-team.github.io/catalyst/api/loggers.html
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
