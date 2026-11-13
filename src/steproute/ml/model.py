"""The LightGBM predictor: training, the saved format, and the ``Predictor`` that serves it.

Needs the ``ml`` extra (``lightgbm``, ``numpy``). ``lightgbm`` is imported inside the functions
that use it, so importing this module costs nothing until a model is trained or loaded.

**Saved format** (a directory, no pickle):

``model.txt``   the booster in LightGBM's text format, trimmed to the best iteration
``meta.json``   format and version, the feature spec (with every column name), the calibrator,
                the LightGBM parameters, the seed, library versions, the data fingerprint and
                the SHA-256 of ``model.txt``
``probes.json`` the probe table (probe task ids in order, and each configuration's results)
``roles.json``  the role of every task (probe, train, tune, calibration, test, excluded)

``TabularModel.load`` refuses a directory whose format or feature version is unknown, whose
``model.txt`` does not match its recorded hash, or whose feature columns differ from what this
code builds (``FeatureSpecMismatch``): it never predicts on columns it cannot vouch for.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from steproute.core.catalog import Catalog
from steproute.core.schema import Action, StepState
from steproute.eval.data import GridData, task_state
from steproute.ml.calibration import (
    Calibrator,
    fit_calibrator,
    log_loss,
    select_calibrator,
    sigmoid,
)
from steproute.ml.data import Dataset, build_dataset, probe_table_from_grid
from steproute.ml.features import (
    FEATURE_VERSION,
    FeatureBuilder,
    FeatureSpec,
    ProbeTable,
    task_feature_keys,
)
from steproute.ml.splits import Fractions, Roles, assign_roles
from steproute.ml.stamp import STAMP, is_synthetic

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "FeatureSpecMismatch",
    "ModelFormatError",
    "TabularModel",
    "TabularPredictor",
    "TrainConfig",
    "TrainResult",
    "lightgbm_params",
    "train",
]

FORMAT = "steproute-tabular"
#: Version of the saved layout. Bump when ``meta.json`` or the file set changes.
FORMAT_VERSION = 1


class ModelFormatError(ValueError):
    """A saved model directory cannot be read as this format."""


class FeatureSpecMismatch(ModelFormatError):
    """The saved feature columns are not the ones this code builds."""


@dataclass(frozen=True)
class TrainConfig:
    """Everything that, with the data, decides the trained model."""

    seed: int = 0
    #: Size of the probe set drawn from the training pool (0: no probe features).
    n_probes: int = 50
    #: Probe counts drawn per training row (probe dropout); None means all probes.
    probe_sizes: tuple[int | None, ...] = (0, 5, 20, None)
    fractions: Fractions = field(default_factory=Fractions)
    #: Longest failed history to build conditional (cascade) rows for. 1 matches the default
    #: candidate depth of 2; 0 trains on fresh rows only.
    max_history: int = 1
    history_cap: int | None = 40
    calibration: str = "auto"
    num_boost_round: int = 300
    early_stopping_rounds: int = 30
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_data_in_leaf: int = 40

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_probes": self.n_probes,
            "probe_sizes": list(self.probe_sizes),
            "fractions": {
                "train": self.fractions.train,
                "tune": self.fractions.tune,
                "calibration": self.fractions.calibration,
                "test": self.fractions.test,
            },
            "max_history": self.max_history,
            "history_cap": self.history_cap,
            "calibration": self.calibration,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
        }


def lightgbm_params(cfg: TrainConfig) -> dict[str, Any]:
    """The LightGBM parameters. Every one that affects determinism is pinned here.

    One thread, the row-wise histogram builder and ``deterministic=True`` make training
    independent of the machine's core count and of thread scheduling; every random stream
    (bagging, feature sampling, data sampling) is seeded from ``cfg.seed``.
    """
    s = int(cfg.seed)
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "lambda_l2": 1.0,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": s,
        "bagging_seed": s + 1,
        "feature_fraction_seed": s + 2,
        "data_random_seed": s + 3,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbosity": -1,
    }


def _versions() -> dict[str, str]:
    import lightgbm

    return {
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class TabularModel:
    """A trained booster, its calibrator, its feature spec and its provenance."""

    def __init__(
        self,
        booster: Any,
        calibrator: Calibrator,
        spec: FeatureSpec,
        meta: dict[str, Any],
        probes: ProbeTable,
        roles: Roles,
    ) -> None:
        self.booster = booster
        self.calibrator = calibrator
        self.spec = spec
        self.meta = meta
        self.probes = probes
        self.roles = roles

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic", False))

    def margins(self, X: NDArray[np.float64]) -> list[float]:
        """Raw booster scores (log-odds), one per row."""
        if len(X) == 0:
            return []
        out = self.booster.predict(X, raw_score=True, num_threads=1)
        return [float(v) for v in cast("NDArray[np.float64]", out).tolist()]

    def raw_probabilities(self, margins: Sequence[float]) -> list[float]:
        return [sigmoid(m) for m in margins]

    def calibrated(self, margins: Sequence[float]) -> list[float]:
        return [min(1.0, max(0.0, self.calibrator(m))) for m in margins]

    # --- saving and loading ---

    def save(self, out: str | Path) -> Path:
        """Write the model directory (creating it) and return its path."""
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        model_file = d / "model.txt"
        model_file.write_text(self.booster.model_to_string(), encoding="utf-8")
        meta = dict(self.meta)
        meta["model_sha256"] = _sha256_file(model_file)
        meta["calibrator"] = self.calibrator.to_dict()
        meta["feature_spec"] = self.spec.to_dict()
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        (d / "probes.json").write_text(
            json.dumps(self.probes.to_dict(), sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (d / "roles.json").write_text(
            json.dumps(self.roles.to_dict(), indent=1, allow_nan=False) + "\n", encoding="utf-8"
        )
        return d

    @classmethod
    def load(cls, path: str | Path) -> TabularModel:
        """Load a saved model, refusing anything that is not exactly this format and these
        feature columns. Raises ``ModelFormatError`` or ``FeatureSpecMismatch``."""
        import lightgbm as lgb

        d = Path(path)
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            probes_d = json.loads((d / "probes.json").read_text(encoding="utf-8"))
            roles_d = json.loads((d / "roles.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelFormatError(f"{d}: cannot read the model files: {exc}") from None
        if not isinstance(meta, dict) or meta.get("format") != FORMAT:
            raise ModelFormatError(f"{d}: not a {FORMAT} model")
        if meta.get("format_version") != FORMAT_VERSION:
            raise ModelFormatError(
                f"{d}: format version {meta.get('format_version')!r}, this code reads "
                f"{FORMAT_VERSION}"
            )
        model_file = d / "model.txt"
        if not model_file.is_file():
            raise ModelFormatError(f"{d}: model.txt is missing")
        if _sha256_file(model_file) != meta.get("model_sha256"):
            raise ModelFormatError(f"{d}: model.txt does not match the hash in meta.json")
        spec_d = meta.get("feature_spec")
        if not isinstance(spec_d, dict):
            raise ModelFormatError(f"{d}: meta.json has no feature_spec")
        spec = FeatureSpec.from_dict(spec_d)
        if spec.version != FEATURE_VERSION:
            raise FeatureSpecMismatch(
                f"{d}: trained with feature version {spec.version}, this code builds version "
                f"{FEATURE_VERSION}; retrain"
            )
        if list(spec.names) != list(spec_d.get("names", [])):
            raise FeatureSpecMismatch(
                f"{d}: the saved column list differs from the columns this code builds for "
                "the same spec; refusing to predict on columns it cannot vouch for"
            )
        booster = lgb.Booster(model_file=str(model_file))
        if list(booster.feature_name()) != list(spec.names):
            raise FeatureSpecMismatch(
                f"{d}: the booster's columns differ from the feature spec in meta.json"
            )
        saved = meta.get("versions", {})
        now = _versions()
        for lib in ("lightgbm", "numpy"):
            if saved.get(lib) != now[lib]:
                warnings.warn(
                    f"{d}: trained with {lib} {saved.get(lib)}, running {now[lib]}",
                    stacklevel=2,
                )
        return cls(
            booster,
            Calibrator.from_dict(meta["calibrator"]),
            spec,
            meta,
            ProbeTable.from_dict(probes_d),
            Roles.from_dict(roles_d),
        )


# --- training -------------------------------------------------------------------------------


@dataclass
class TrainResult:
    model: TabularModel
    summary: dict[str, Any]


def _fit_booster(
    cfg: TrainConfig, train: Dataset, tune: Dataset, names: Sequence[str]
) -> tuple[Any, int]:
    import lightgbm as lgb

    params = lightgbm_params(cfg)
    d_train = lgb.Dataset(train.X, label=train.y, feature_name=list(names), free_raw_data=False)
    valid = []
    callbacks: list[Callable[..., Any]] = []
    if len(tune) > 0 and cfg.early_stopping_rounds > 0:
        valid = [lgb.Dataset(tune.X, label=tune.y, reference=d_train, feature_name=list(names))]
        callbacks = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
    booster = lgb.train(
        params,
        d_train,
        num_boost_round=cfg.num_boost_round,
        valid_sets=valid,
        callbacks=callbacks,
    )
    best = booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()
    # Trim to the best iteration and reload, so the live model and a loaded one are the same.
    text = booster.model_to_string(num_iteration=best)
    return lgb.Booster(model_str=text), int(best)


def train(
    grid: GridData,
    catalog: Catalog,
    cfg: TrainConfig | None = None,
    *,
    holdout_models: Sequence[str] = (),
    group_feature: str | None = None,
    source_name: str | None = None,
) -> TrainResult:
    """Train, calibrate and package a model from a replay grid.

    Tasks get roles first (``assign_roles``): the probe set is drawn out, then train, tune
    (early stopping), calibration and test are dealt by task (or by ``group_feature`` when
    given, keeping each stratum whole). The trees see only ``train`` (and ``tune`` for the
    stopping round); the calibrator is fitted on ``calibration`` alone; ``test`` is not read.
    ``holdout_models`` gives no training rows for those models, but their probe results still
    describe them (the cold-start setting).
    """
    cfg = cfg or TrainConfig()
    missing = sorted({m for m, _ in grid.configs() if m not in catalog})
    if missing:
        raise ValueError(f"the grid has models that the catalog does not: {missing}")

    group_of = None
    if group_feature is not None:

        def group_of(t: str) -> str:
            return str(task_state(grid.tasks[t]).features.get(group_feature, t))

    roles = assign_roles(
        sorted(grid.tasks),
        seed=cfg.seed,
        n_probe=cfg.n_probes,
        fractions=cfg.fractions,
        group_of=group_of,
        grid_splits={t: grid.tasks[t].split for t in grid.tasks},
    )
    probes = probe_table_from_grid(grid, roles.probe_order)
    train_tasks = roles.tasks("train")
    spec = FeatureSpec(task_keys=task_feature_keys(task_state(grid.tasks[t]) for t in train_tasks))
    builder = FeatureBuilder(spec, catalog, probes)

    def table(role: str) -> Dataset:
        return build_dataset(
            grid,
            roles.tasks(role),
            catalog,
            builder,
            seed=cfg.seed,
            max_history=cfg.max_history,
            history_cap=cfg.history_cap,
            probe_sizes=cfg.probe_sizes if probes.order else None,
            exclude_models=holdout_models,
        )

    d_train, d_tune, d_cal = table("train"), table("tune"), table("calibration")
    if len(d_train) == 0:
        raise ValueError("no training rows: the grid has no labelled runs in the train role")
    booster, best = _fit_booster(cfg, d_train, d_tune, spec.names)
    model = TabularModel(booster, Calibrator(), spec, {}, probes, roles)
    cal_margins = model.margins(d_cal.X)
    cal_y = [int(v) for v in d_cal.y.tolist()]
    cv_losses: dict[str, float] = {}
    if cfg.calibration == "auto":
        calibrator, cv_losses = select_calibrator(cal_margins, cal_y, d_cal.task_ids, seed=cfg.seed)
    else:
        calibrator = fit_calibrator(cfg.calibration, cal_margins, cal_y)
    model.calibrator = calibrator

    synthetic = is_synthetic(grid.header)
    params = lightgbm_params(cfg)
    data = {
        "source": source_name or Path(grid.source).name,
        "grid_sha256": grid.sha256,
        "n_tasks": len(grid.tasks),
        "n_runs": len(grid.runs),
        "roles": roles.counts(),
        "roles_fingerprint": roles.fingerprint(),
        "roles_source": roles.source,
        "probes_fingerprint": probes.fingerprint(),
        "holdout_models": sorted(holdout_models),
        "rows": {"train": len(d_train), "tune": len(d_tune), "calibration": len(d_cal)},
    }
    fingerprint = hashlib.sha256(
        _canonical(
            {
                "data": data,
                "spec": spec.fingerprint(),
                "config": cfg.to_dict(),
                "params": params,
            }
        ).encode()
    ).hexdigest()
    model.meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "seed": cfg.seed,
        "synthetic": synthetic,
        "stamp": STAMP if synthetic else None,
        "train_config": cfg.to_dict(),
        "lightgbm_params": params,
        "best_iteration": best,
        "calibration_cv_log_loss": cv_losses,
        "versions": _versions(),
        "data": data,
        "data_fingerprint": fingerprint,
    }
    summary: dict[str, Any] = {
        "rows": data["rows"],
        "roles": data["roles"],
        "best_iteration": best,
        "calibration": calibrator.kind,
        "calibration_cv_log_loss": cv_losses,
        "calibration_fit_log_loss_before": (
            log_loss(model.raw_probabilities(cal_margins), cal_y) if cal_y else None
        ),
        "calibration_fit_log_loss_after": (
            log_loss(model.calibrated(cal_margins), cal_y) if cal_y else None
        ),
        "data_fingerprint": fingerprint,
        "synthetic": synthetic,
    }
    return TrainResult(model, summary)


# --- serving --------------------------------------------------------------------------------


class TabularPredictor:
    """``P(success | state, step)`` from a trained model, for ``ScoredPolicy``.

    Satisfies ``steproute.core.policy.Predictor`` (``predict``) and offers ``predict_many``,
    which ``ScoredPolicy`` uses when present: one feature matrix and one booster call per
    decision instead of one per candidate. ``predict`` is ``predict_many`` of one pair, so the
    two paths cannot differ.

    ``probes`` defaults to the model's own probe table; pass another to describe a new model
    from its own probe results. ``n_probes`` limits how many probes are used (0: none, the
    cold-start representation; None: all). ``calibrated=False`` returns the booster's own
    probability, to measure what calibration changes.
    """

    def __init__(
        self,
        model: TabularModel,
        catalog: Catalog,
        probes: ProbeTable | None = None,
        *,
        n_probes: int | None = None,
        calibrated: bool = True,
    ) -> None:
        self.model = model
        self.calibrated = calibrated
        self.n_probes = n_probes
        self.builder = FeatureBuilder(
            model.spec, catalog, probes if probes is not None else model.probes, n_probes
        )
        self.name = "tabular" if calibrated else "tabular-uncalibrated"

    def predict_many(self, pairs: Sequence[tuple[StepState, Action]]) -> list[float]:
        if not pairs:
            return []
        margins = self.model.margins(self.builder.matrix(pairs))
        if self.calibrated:
            return self.model.calibrated(margins)
        return self.model.raw_probabilities(margins)

    def predict(self, state: StepState, action_step: Action) -> float:
        return self.predict_many([(state, action_step)])[0]
