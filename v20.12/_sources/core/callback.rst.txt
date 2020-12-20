Callback
==============================================================================

Callback — an abstraction that lets you **customize** your experiment run logic.
To give users maximum flexibility and extensibility Catalyst supports
callback execution anywhere in the training loop.

.. image:: https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master/third_party_pics/catalyst102-callback.png
    :alt: Callback

Such a callback system allows you to quickly
enable/disable metrics and other dl tricks,
like gradient accumulation, mixup, batch overfitting, early stopping —
with only a few lines of code. During any experiment run.

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
