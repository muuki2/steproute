"""Code adapted from upstream routers, each file under its own licence.

- ``graphrouter_gnn.py``: from LLMRouter (MIT); the notice is at the top of the file and in
  ``LICENSE-LLMRouter``.
- ``routellm_mf.py``: from RouteLLM (Apache-2.0); the notice is at the top of the file, the
  licence text is ``LICENSE-RouteLLM``.

Both need torch (and the first torch-geometric): the ``baselines`` extra. Nothing imports them
unless a baseline that uses them is run. Commits and changes are listed in each file and in
``docs/BASELINES.md``.
"""
