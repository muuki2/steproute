# steproute

**Status: pre-alpha. Only the repository scaffold exists; none of the features described below are implemented yet.**

steproute is being built as an open router that picks, for each step of an LLM agent run, which model and effort level to use. The decision is `argmin expected_cost subject to P(success) >= tau`: cost is computed from prices, token counts and cache state, while the probability of success is learned from recorded step outcomes. The plan includes a harness-neutral trace schema and collectors, a cost model, rule-based and learned predictors (LightGBM, then a graph model that can handle new models without retraining), and an evaluator that compares them on the same traces.

See [docs/DESIGN.md](docs/DESIGN.md) for the design and [docs/ISSUES.md](docs/ISSUES.md) for the work plan. The core package (`steproute.core`) will stay free of third-party runtime dependencies.

Licensed under the MIT License. See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved.
