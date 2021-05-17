Callback
==============================================================================

The ``Callback`` is an abstraction that helps you to customize the logic during your run.
Once again, you could do anything natively with PyTorch and Catalyst as a for-loop wrapper.
However, thanks to the callbacks, it's much easier to reuse typical deep learning extensions
like metrics or augmentation tricks.
For example, it's much more convenient to define the required metrics:

- `ML - multiclass classification`_.
- `ML – RecSys`_.

The Callback API is very straightforward and repeats main for-loops in our train-loop abstraction:

.. image:: https://raw.githubusercontent.com/Scitator/catalyst21-post-pics/main/code_callback21.png
    :alt: Runner

You could find a large variety of supported callbacks under the `Callback API section`_.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`ML - multiclass classification`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`ML – RecSys`: https://github.com/catalyst-team/catalyst#minimal-examples
.. _`Callback API section`: https://catalyst-team.github.io/catalyst/api/callbacks.html
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
