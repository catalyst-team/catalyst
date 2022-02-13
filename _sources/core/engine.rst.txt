Engine
==============================================================================

``Engine`` is the main force of the ``Runner``.
It defines the logic of hardware communication and different deep learning techniques usage
like distributed or half-precision training:

.. image:: https://raw.githubusercontent.com/Scitator/catalyst22-post-pics/main/engine.png
    :alt: Engine


Thanks to the ``Engine`` design,
it’s straightforward to adapt your pipeline for different hardware accelerators.
For example, you could easily support:

- `PyTorch distributed setup`_.
- `AMP distributed setup`_.
- `DeepSpeed distributed setup`_.
- `TPU distributed setup`_.

If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`PyTorch distributed setup`: https://catalyst-team.github.io/catalyst/api/engines.html
.. _`AMP distributed setup`: https://catalyst-team.github.io/catalyst/api/engines.html
.. _`DeepSpeed distributed setup`: https://catalyst-team.github.io/catalyst/api/engines.html
.. _`TPU distributed setup`: https://catalyst-team.github.io/catalyst/api/engines.html
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
