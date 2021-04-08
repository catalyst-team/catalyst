Engine
==============================================================================

``Engine`` is the main force of the ``Runner``.
It defines the logic of hardware communication and different deep learning techniques usage
like distributed or half-precision training:

.. image:: https://raw.githubusercontent.com/Scitator/catalyst21-post-pics/main/code_engine21.png
    :alt: Engine


Thanks to the ``Engine`` design,
it’s straightforward to adapt your pipeline for different hardware accelerators.
For example, you could easily support:

- `PyTorch distribute setup`_.
- `Nvidia-Apex setup`_.
- `AMP distributed setup`_.

Moreover, we are working on other hardware accelerators support like DeepSpeed, Horovod, or TPU.
You could watch the progress of engine development under the `Engine API section`_.


If you haven't found the answer for your question, feel free to `join our slack`_ for the discussion.

.. _`PyTorch distribute setup`: https://catalyst-team.github.io/catalyst/api/engines.html#distributeddataparallelampengine
.. _`Nvidia-Apex setup`: https://catalyst-team.github.io/catalyst/api/engines.html#distributeddataparallelapexengine
.. _`AMP distributed setup`: https://catalyst-team.github.io/catalyst/api/engines.html#distributeddataparallelampengine
.. _`Engine API section`: https://catalyst-team.github.io/catalyst/api/engines.html
.. _`join our slack`: https://join.slack.com/t/catalyst-team-core/shared_invite/zt-d9miirnn-z86oKDzFMKlMG4fgFdZafw
