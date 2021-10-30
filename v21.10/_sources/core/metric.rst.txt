Metric
==============================================================================

Catalyst API also involves ``Metric`` abstraction
for convenient metric computation during an experiment run. Its API is quite simple:

.. image:: https://raw.githubusercontent.com/Scitator/catalyst21-post-pics/main/code_metric21.png
    :alt: Runner


You could find all the supported metrics under the `Metric API section`_.

Catalyst Metric API has a default `update` and `compute` methods
to support per-batch statistic accumulation and final computation during training.
All metrics also support `update` and `compute` key-value extensions
for convenient usage during the run — it gives you the flexibility
to store any number of metrics or aggregations you want
with a simple communication protocol to use for their logging.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.


.. _`Metric API section`: https://catalyst-team.github.io/catalyst/api/metrics.html#runner-metrics
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
