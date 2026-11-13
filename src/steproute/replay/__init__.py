"""The replay runner (issue 18): the dataset's grid of task runs under fixed configurations.

See ``docs/REPLAY.md``. The modules:

* ``config``: the strict TOML config, with the safe defaults (caps, per-run limits).
* ``plan``: the named plans (``pilot``, ``sparse``) built from the manifest and the catalog,
  in cheapest-first order.
* ``ledger``: the append-only spend ledger and the cap guard.
* ``journal``: the run journal (resume) and the output-directory lock.
* ``agent``: the bash agent scaffold; ``workspace``: where its commands run.
* ``proxyrun``: one run's private proxy and the chat client that enforces every limit.
* ``runner``: the orchestrator; ``report``: the pilot report; ``preflight``: what is missing.

Importing this package makes no network call and needs no extra; a run that calls a model
needs the ``[proxy]`` extra.
"""
