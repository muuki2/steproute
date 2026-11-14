"""External baselines (issue 22): published routers, run under our evaluator on our data.

Each adapter trains (or configures) one published router on the training tasks of a replay grid
and exposes it as an evaluator ``TargetPolicy``:

- ``steproute.baselines.paretobandit``: ParetoBandit (LinUCB with a cost penalty), through the
  ``paretobandit`` package.
- ``steproute.baselines.graphrouter``: LLMRouter's ``graphrouter`` (a GNN over a query-LLM
  graph), adapted from LLMRouter's source (``_vendor/graphrouter_gnn.py``, MIT).
- ``steproute.baselines.routellm``: RouteLLM's matrix-factorisation router in its two-model
  setting, adapted from RouteLLM's source (``_vendor/routellm_mf.py``, Apache-2.0).

OpenSquilla's LightGBM router cannot be run fairly here; ``docs/BASELINES.md`` says why.

The heavy libraries (``paretobandit``, ``torch``, ``torch-geometric``) are the ``baselines``
extra and are imported only inside the functions that need them. This package and
``steproute.baselines.common`` import with numpy alone. Versions, commits, licences and the
fairness notes are in ``docs/BASELINES.md``; ``python -m steproute.baselines`` writes a report.
"""

from __future__ import annotations

__all__: list[str] = []
