#!/usr/bin/env python
"""The phase2ML learner roster: three heads, one interface each.

Every learner here is optional. A missing package removes exactly one row from
the leaderboard and is recorded as SKIPPED with the package name -- it never
fails the run, because a bake-off that only works when all ten libraries are
installed is a bake-off nobody runs.

Interfaces
----------
occurrence  scikit-learn estimator: ``fit(X, y)`` and ``predict_proba(X)``.
magnitude   ``fit(X, y_log)`` and ``predict_log_quantiles(X, qs) -> (n, len(qs))``
            IN LOG1P SPACE. The driver exponentiates once, exactly the way
            ``phase2_train.magnitude_quantiles`` does, so CRPS is computed on
            the same scale for every row of the leaderboard.
duration    ``fit(X, duration_hours, observed)`` and
            ``predict_quantiles(X, qs) -> (n, len(qs))`` in hours.

The magnitude head is where a naive bake-off goes wrong. CRPS scores a
DISTRIBUTION, and most tabular regressors emit a point. Handing a point
prediction to a proper scoring rule -- or dressing it with one global residual
spread -- does not measure the learner, it measures the dressing. So each
magnitude learner declares how its distribution is produced:

  native_quantile   the learner is fitted once per reported quantile under
                    pinball loss (LightGBM, sklearn GBM, XGBoost)
  multi_quantile    one fit emits all quantiles (CatBoost MultiQuantile)
  quantile_forest   the empirical distribution of training targets landing in
                    the same leaves (Meinshausen's QRF, sampled)
  parametric        a fitted conditional density (NGBoost)
  residual_dressed  a point learner plus LEVEL-CONDITIONAL residual quantiles
                    estimated out-of-fold on the training window

``distribution_route`` travels with every score. A residual-dressed MLP that
edges out NGBoost on CRPS is a different claim from a natively quantile-fitted
model that does, and the leaderboard has to make that visible.
"""
from __future__ import annotations

import importlib.util as iu
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.common.logio import get_logger

log = get_logger("phase2ML.models")


def have(package: str) -> bool:
    return iu.find_spec(package) is not None


def missing(packages: tuple[str, ...]) -> list[str]:
    return [p for p in packages if not have(p)]


@dataclass
class Learner:
    """One leaderboard row's worth of recipe."""

    name: str
    requires: tuple[str, ...]
    build: object                     # callable(cfg, seed) -> model
    family: str = "tree"
    route: str = ""
    grid: dict = field(default_factory=dict)
    note: str = ""
    is_incumbent: bool = False
    censoring: str = "n/a"
    # scikit-learn's MLPClassifier has no sample_weight, so the driver cannot
    # apply the positive-class weighting every other occurrence learner gets.
    # Recorded rather than hidden: an unweighted learner on a 3%-positive
    # target is at a real disadvantage, and that belongs in the leaderboard.
    supports_sample_weight: bool = True


# =============================================================================
# Occurrence
# =============================================================================

def _lgbm_occ(cfg, seed):
    import lightgbm as lgb
    # Deliberately the incumbent's exact hyperparameters (phase2_train
    # .lgb_classifier). This row exists to prove the harness is fair: if it
    # does not reproduce the published incumbent Brier to within the noise of
    # the scale_pos_weight computation, something in phase2ML differs from the
    # incumbent pipeline and every other row is suspect.
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=int(cfg["n_boost_rounds"]),
        learning_rate=0.05, num_leaves=15, min_child_samples=30,
        subsample=0.9, colsample_bytree=0.9,
        random_state=seed, n_jobs=-1, verbose=-1)


def _rf_occ(cfg, seed):
    from sklearn.ensemble import RandomForestClassifier
    # No class_weight here: positive-class weighting is applied uniformly by
    # the driver as a sample_weight, so every occurrence learner sees the same
    # imbalance correction. Setting it in two places would double-weight this
    # row and quietly make the forests look better than the boosters.
    return RandomForestClassifier(
        n_estimators=int(cfg.get("forest_trees", 400)), min_samples_leaf=20,
        max_features="sqrt", random_state=seed, n_jobs=-1)


def _et_occ(cfg, seed):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=int(cfg.get("forest_trees", 400)), min_samples_leaf=20,
        max_features="sqrt", random_state=seed, n_jobs=-1)


def _hgb_occ(cfg, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=int(cfg["n_boost_rounds"]), learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def _xgb_occ(cfg, seed):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=int(cfg["n_boost_rounds"]), learning_rate=0.05,
        max_depth=4, min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1)


def _cat_occ(cfg, seed):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=int(cfg["n_boost_rounds"]), learning_rate=0.05, depth=5,
        l2_leaf_reg=3.0, loss_function="Logloss", random_seed=seed,
        verbose=False, allow_writing_files=False)


def _mlp_occ(cfg, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # Scaling is not optional here: the feature block mixes m/s, millimetres,
    # percentages and a 0-3650 day counter, and an unscaled MLP on that is a
    # measurement of the units, not of the model class.
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=512,
            learning_rate_init=1e-3, max_iter=300, early_stopping=True,
            n_iter_no_change=15, random_state=seed)),
    ])


def _enet_occ(cfg, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # The ML-side skill floor, on the FULL feature set -- distinct from the
    # incumbent's deliberately reduced 9-feature statsmodels GLM. If the
    # boosted models cannot beat a penalised linear model on the same 26
    # features, that is the headline finding, not a footnote.
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
            max_iter=2000, random_state=seed)),
    ])


OCCURRENCE: list[Learner] = [
    Learner("lightgbm_incumbent", ("lightgbm",), _lgbm_occ, "tree",
            is_incumbent=True,
            note="incumbent Phase 2 configuration, refitted inside this harness "
                 "as the fairness check", grid={}),
    Learner("random_forest", ("sklearn",), _rf_occ, "tree",
            grid={"min_samples_leaf": [10, 20, 40]}),
    Learner("extra_trees", ("sklearn",), _et_occ, "tree",
            grid={"min_samples_leaf": [10, 20, 40]}),
    Learner("hist_gradient_boosting", ("sklearn",), _hgb_occ, "tree",
            grid={"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    Learner("xgboost", ("xgboost",), _xgb_occ, "tree",
            grid={"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1]}),
    Learner("catboost", ("catboost",), _cat_occ, "tree",
            grid={"depth": [4, 5, 7], "learning_rate": [0.03, 0.05, 0.1]}),
    Learner("mlp", ("sklearn",), _mlp_occ, "neural",
            grid={"mlp__alpha": [1e-4, 1e-3, 1e-2],
                  "mlp__hidden_layer_sizes": [(64, 32), (128, 64)]},
            supports_sample_weight=False,
            note="no sample_weight support in MLPClassifier, so this row is "
                 "fitted WITHOUT the positive-class weighting the others get"),
    Learner("elasticnet_logistic", ("sklearn",), _enet_occ, "linear",
            grid={"lr__C": [0.1, 0.5, 2.0]},
            note="ML-side skill floor on the full feature set"),
]


# =============================================================================
# Magnitude -- every learner must emit a predictive DISTRIBUTION
# =============================================================================

def _monotone(log_q: np.ndarray) -> np.ndarray:
    """Repair quantile crossing by rearrangement, the incumbent's convention.

    Independently fitted quantile regressions are not guaranteed monotone, and
    a crossed quantile function makes CRPS meaningless rather than merely
    worse. Sorting each row is the standard rearrangement fix; the driver
    records how often it was needed, because a model that crosses constantly
    is straining and that is a finding about the model.
    """
    return np.sort(np.asarray(log_q, dtype=float), axis=1)


class PerQuantileGBM:
    """One pinball-loss fit per reported quantile. route=native_quantile."""

    route = "native_quantile"

    def __init__(self, factory, quantiles):
        self._factory = factory
        self.quantiles = [float(q) for q in quantiles]
        self.models: dict[float, object] = {}
        self.n_crossings = 0

    def fit(self, X, y_log):
        for q in self.quantiles:
            model = self._factory(q)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y_log)
            self.models[q] = model
        return self

    def predict_log_quantiles(self, X, qs):
        trained = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        self.n_crossings += int((np.diff(trained, axis=1) < 0).any(axis=1).sum())
        return _interp(_monotone(trained), self.quantiles, qs)


class MultiQuantileModel:
    """A single fit that emits every quantile. route=multi_quantile."""

    route = "multi_quantile"

    def __init__(self, factory, quantiles):
        self._factory = factory
        self.quantiles = [float(q) for q in quantiles]
        self.model = None

    def fit(self, X, y_log):
        self.model = self._factory(self.quantiles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y_log)
        return self

    def predict_log_quantiles(self, X, qs):
        raw = np.asarray(self.model.predict(X), dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        return _interp(_monotone(raw), self.quantiles, qs)


class QuantileForest:
    """Meinshausen's quantile regression forest. route=quantile_forest.

    A random forest's per-tree predictions are the spread of the conditional
    MEAN estimate, which is far narrower than the conditional distribution of
    the target. Scoring that with CRPS would make every forest look wildly
    overconfident for a reason that is an artefact of the read-out, not a
    property of the model. The correct predictive distribution is the empirical
    distribution of the TRAINING targets that land in the same leaves.

    Computed by sampling rather than by exact weighting: for each tree, draw
    ``draws_per_tree`` training targets from the leaf the row falls into, pool
    them across trees, and take empirical quantiles of the pool. With 400 trees
    that is 3200 draws per row -- ample for seven reported quantiles -- and it
    is O(n_rows x n_trees) instead of the O(n_rows x n_train x n_trees) an
    exact weighted quantile costs. The draw is seeded, so the result is
    reproducible.
    """

    route = "quantile_forest"

    def __init__(self, factory, seed: int, draws_per_tree: int = 8):
        self._factory = factory
        self.seed = int(seed)
        self.draws_per_tree = int(draws_per_tree)
        self.model = None
        self._leaf_targets: list[dict[int, np.ndarray]] = []

    def fit(self, X, y_log):
        self.model = self._factory()
        self.model.fit(X, y_log)
        leaves = self.model.apply(np.asarray(X, dtype=float))
        y = np.asarray(y_log, dtype=float)
        self._leaf_targets = []
        for t in range(leaves.shape[1]):
            column = leaves[:, t]
            order = np.argsort(column, kind="stable")
            sorted_leaf = column[order]
            bounds = np.searchsorted(sorted_leaf, np.unique(sorted_leaf), side="left")
            bounds = np.append(bounds, len(sorted_leaf))
            table = {}
            uniq = np.unique(sorted_leaf)
            for i, leaf_id in enumerate(uniq):
                table[int(leaf_id)] = y[order[bounds[i]:bounds[i + 1]]]
            self._leaf_targets.append(table)
        return self

    def predict_log_quantiles(self, X, qs):
        leaves = self.model.apply(np.asarray(X, dtype=float))
        rng = np.random.default_rng(self.seed)
        n_rows, n_trees = leaves.shape
        pool = np.empty((n_rows, n_trees * self.draws_per_tree), dtype=float)
        for t in range(n_trees):
            table = self._leaf_targets[t]
            column = leaves[:, t]
            for row in range(n_rows):
                values = table.get(int(column[row]))
                start = t * self.draws_per_tree
                if values is None or values.size == 0:
                    pool[row, start:start + self.draws_per_tree] = np.nan
                else:
                    pool[row, start:start + self.draws_per_tree] = rng.choice(
                        values, size=self.draws_per_tree, replace=True)
        return np.nanquantile(pool, np.asarray(qs, dtype=float), axis=1).T


class ResidualDressed:
    """Point learner + LEVEL-CONDITIONAL residual quantiles. route=residual_dressed.

    The lazy version of this dresses every prediction with the same residual
    spread, which is wrong in a way that matters here: customer-hours span four
    to five orders of magnitude and the residual spread grows with the
    predicted level, so one global spread makes the small events look
    hopelessly uncertain and the large ones overconfident -- and CRPS punishes
    both. Residual quantiles are therefore estimated within bins of the
    predicted level.

    Residuals come from OUT-OF-FOLD training predictions, never in-sample ones.
    In-sample residuals of a boosted model are a fraction of its true error, so
    dressing with them produces a spuriously sharp distribution that scores
    well until it meets held-out data.
    """

    route = "residual_dressed"

    def __init__(self, factory, n_bins: int = 5, cv_folds: int = 4,
                 seed: int = 0):
        self._factory = factory
        self.n_bins = int(n_bins)
        self.cv_folds = int(cv_folds)
        self.seed = int(seed)
        self.model = None
        self.edges: np.ndarray | None = None
        self.residual_q: np.ndarray | None = None
        self.grid = np.linspace(0.005, 0.995, 199)

    def fit(self, X, y_log, groups=None):
        from sklearn.model_selection import GroupKFold, KFold

        X = np.asarray(X, dtype=float)
        y = np.asarray(y_log, dtype=float)
        oof = np.full(len(y), np.nan)
        if groups is not None and pd.Series(groups).nunique() >= self.cv_folds:
            splitter = GroupKFold(n_splits=self.cv_folds).split(X, y, groups)
        else:
            splitter = KFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.seed).split(X)
        for fit_idx, hold_idx in splitter:
            fold = self._factory()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold.fit(X[fit_idx], y[fit_idx])
            oof[hold_idx] = fold.predict(X[hold_idx])
        ok = ~np.isnan(oof)
        residual = y[ok] - oof[ok]
        level = oof[ok]

        bins = min(self.n_bins, max(1, len(level) // 30))
        quantile_edges = np.quantile(level, np.linspace(0, 1, bins + 1))
        quantile_edges[0], quantile_edges[-1] = -np.inf, np.inf
        self.edges = np.unique(quantile_edges)
        index = np.clip(np.searchsorted(self.edges, level, side="right") - 1,
                        0, len(self.edges) - 2)
        self.residual_q = np.vstack([
            np.quantile(residual[index == b], self.grid)
            if (index == b).sum() >= 10 else np.quantile(residual, self.grid)
            for b in range(len(self.edges) - 1)])

        self.model = self._factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        return self

    def predict_log_quantiles(self, X, qs):
        centre = np.asarray(self.model.predict(np.asarray(X, dtype=float)),
                            dtype=float)
        index = np.clip(np.searchsorted(self.edges, centre, side="right") - 1,
                        0, len(self.edges) - 2)
        # Interpolate each bin's residual quantile curve onto the requested
        # probabilities once, then shift by each row's point prediction.
        wanted = np.asarray(qs, dtype=float)
        per_bin = np.vstack([np.interp(wanted, self.grid, self.residual_q[b])
                             for b in range(self.residual_q.shape[0])])
        return _monotone(centre[:, None] + per_bin[index])


class NGBoostDensity:
    """The incumbent magnitude model, refitted here. route=parametric."""

    route = "parametric"

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, int(seed)
        self.model = None

    def fit(self, X, y_log):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        self.model = NGBRegressor(
            Dist=Normal, n_estimators=int(self.cfg["n_boost_rounds"]),
            learning_rate=0.03, verbose=False, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(np.asarray(X, dtype=float), np.asarray(y_log, dtype=float))
        return self

    def predict_log_quantiles(self, X, qs):
        dist = self.model.pred_dist(np.asarray(X, dtype=float))
        return _monotone(np.column_stack([dist.ppf(float(q)) for q in qs]))


def _interp(base: np.ndarray, trained: list[float], wanted) -> np.ndarray:
    """Reuse the incumbent's probit-slope interpolation/extrapolation exactly."""
    from src.phase2_train import _interp_log_quantiles
    wanted = [float(q) for q in wanted]
    if list(trained) == wanted:
        return base
    return _interp_log_quantiles(base, [float(q) for q in trained], wanted)


def _mag_lgbm(cfg, seed):
    import lightgbm as lgb

    def factory(q):
        return lgb.LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=int(cfg["n_boost_rounds"]),
            learning_rate=0.04, num_leaves=15, min_child_samples=20,
            random_state=seed, n_jobs=-1, verbose=-1)
    return PerQuantileGBM(factory, cfg["quantiles"])


def _mag_sk_gbm(cfg, seed):
    from sklearn.ensemble import GradientBoostingRegressor

    def factory(q):
        return GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=int(cfg.get("sk_gbm_rounds", 300)),
            learning_rate=0.05, max_depth=3, min_samples_leaf=20,
            random_state=seed)
    return PerQuantileGBM(factory, cfg["quantiles"])


def _mag_xgb(cfg, seed):
    import xgboost as xgb

    def factory(q):
        return xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=float(q),
            n_estimators=int(cfg["n_boost_rounds"]), learning_rate=0.04,
            max_depth=4, min_child_weight=10, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist", random_state=seed, n_jobs=-1)
    return PerQuantileGBM(factory, cfg["quantiles"])


def _mag_catboost(cfg, seed):
    from catboost import CatBoostRegressor

    def factory(quantiles):
        alphas = ",".join(f"{float(q):.4g}" for q in quantiles)
        return CatBoostRegressor(
            loss_function=f"MultiQuantile:alpha={alphas}",
            iterations=int(cfg["n_boost_rounds"]), learning_rate=0.05, depth=5,
            random_seed=seed, verbose=False, allow_writing_files=False)
    return MultiQuantileModel(factory, cfg["quantiles"])


def _mag_qrf(cfg, seed):
    from sklearn.ensemble import RandomForestRegressor

    def factory():
        return RandomForestRegressor(
            n_estimators=int(cfg.get("forest_trees", 400)), min_samples_leaf=20,
            max_features="sqrt", random_state=seed, n_jobs=-1)
    return QuantileForest(factory, seed, int(cfg.get("qrf_draws_per_tree", 8)))


def _mag_mlp(cfg, seed):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def factory():
        return Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=256,
                learning_rate_init=1e-3, max_iter=400, early_stopping=True,
                n_iter_no_change=15, random_state=seed)),
        ])
    return ResidualDressed(factory, seed=seed)


def _mag_ngboost(cfg, seed):
    return NGBoostDensity(cfg, seed)


MAGNITUDE: list[Learner] = [
    Learner("ngboost_incumbent", ("ngboost",), _mag_ngboost, "boost",
            route="parametric", is_incumbent=True,
            note="incumbent magnitude model, refitted inside this harness"),
    Learner("lightgbm_quantile", ("lightgbm",), _mag_lgbm, "boost",
            route="native_quantile",
            note="the incumbent's documented fallback route"),
    Learner("sklearn_gbm_quantile", ("sklearn",), _mag_sk_gbm, "boost",
            route="native_quantile"),
    Learner("xgboost_quantile", ("xgboost",), _mag_xgb, "boost",
            route="native_quantile",
            note="needs xgboost >= 2.0 for reg:quantileerror"),
    Learner("catboost_multiquantile", ("catboost",), _mag_catboost, "boost",
            route="multi_quantile"),
    Learner("quantile_random_forest", ("sklearn",), _mag_qrf, "tree",
            route="quantile_forest"),
    Learner("mlp_residual", ("sklearn",), _mag_mlp, "neural",
            route="residual_dressed",
            note="point net dressed with out-of-fold, level-conditional residuals"),
]


# =============================================================================
# Duration -- right-censored restoration time
# =============================================================================
#
# Roughly a fifth of observed restorations are right-censored (the outage was
# still running when the record window closed). Handling that is not a detail:
# a learner that treats a censored duration as an observed one is systematically
# trained to under-predict long restorations, which are precisely the ones the
# decision analysis in section 9 is about. Every learner below declares how it
# treats censoring, and one deliberately ignores it so the leaderboard can
# quantify what that costs rather than asserting it.


def _survival_quantiles(times: np.ndarray, survival: np.ndarray,
                        qs) -> np.ndarray:
    """Read quantiles off a step survival curve S(t) = P(T > t).

    The q-th quantile of T is the first time where S(t) <= 1 - q. Under heavy
    censoring the curve can plateau above 1 - q and never reach it; the honest
    read-out there is "at least the last observed time", so the largest
    evaluated time is returned and the caller records how often that happened.
    Silently returning the median instead would invent restoration times that
    the data does not support.
    """
    times = np.asarray(times, dtype=float)
    survival = np.asarray(survival, dtype=float)
    out = np.empty((survival.shape[0], len(qs)), dtype=float)
    truncated = 0
    for j, q in enumerate(qs):
        target = 1.0 - float(q)
        reached = survival <= target
        idx = np.where(reached.any(axis=1), reached.argmax(axis=1), len(times) - 1)
        truncated += int((~reached.any(axis=1)).sum())
        out[:, j] = times[idx]
    _survival_quantiles.last_truncated = truncated
    return np.sort(out, axis=1)


_survival_quantiles.last_truncated = 0


class SksurvModel:
    """Any scikit-survival estimator with predict_survival_function."""

    censoring = "modelled"

    def __init__(self, factory):
        self._factory = factory
        self.model = None
        self.truncated_fraction = 0.0

    @staticmethod
    def _structured(duration, observed):
        return np.array(list(zip(np.asarray(observed).astype(bool),
                                 np.asarray(duration, dtype=float))),
                        dtype=[("event", "?"), ("time", "<f8")])

    def fit(self, X, duration, observed):
        self.model = self._factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(np.asarray(X, dtype=float),
                           self._structured(duration, observed))
        return self

    def predict_quantiles(self, X, qs):
        functions = self.model.predict_survival_function(np.asarray(X, dtype=float))
        times = np.asarray(functions[0].x, dtype=float)
        survival = np.vstack([f(times) for f in functions])
        out = _survival_quantiles(times, survival, qs)
        self.truncated_fraction = (
            _survival_quantiles.last_truncated / max(survival.shape[0] * len(qs), 1))
        return out


class XGBAft:
    """XGBoost's accelerated-failure-time objective. censoring = modelled.

    XGBoost takes interval-censored labels directly: an observed restoration is
    the degenerate interval [t, t] and a censored one is [t, +inf). That makes
    it the only gradient-boosted learner in the roster that uses the censored
    rows as the information they are rather than dropping or mislabelling them.

    Quantiles come from the fitted AFT error distribution: log T = f(x) + sigma*Z
    with Z normal, so the q-th quantile is exp(f(x) + sigma*Phi^-1(q)). sigma is
    the ``aft_loss_distribution_scale`` the model was fitted with, so the spread
    is homoscedastic in log time -- a real limitation, and the reason this row
    reports a spread-skill ratio alongside its concordance.
    """

    censoring = "modelled"

    def __init__(self, cfg, seed, scale: float = 1.0):
        self.cfg, self.seed, self.scale = cfg, int(seed), float(scale)
        self.model = None

    def fit(self, X, duration, observed):
        import xgboost as xgb

        X = np.asarray(X, dtype=float)
        duration = np.asarray(duration, dtype=float)
        observed = np.asarray(observed).astype(bool)
        lower = duration.copy()
        upper = np.where(observed, duration, np.inf)
        dtrain = xgb.DMatrix(X)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": self.scale,
            "tree_method": "hist", "learning_rate": 0.05, "max_depth": 4,
            "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9,
            "seed": self.seed,
        }
        self.model = xgb.train(params, dtrain,
                               num_boost_round=int(self.cfg.get("aft_rounds", 300)))
        return self

    def predict_quantiles(self, X, qs):
        import xgboost as xgb
        from scipy.stats import norm

        centre = np.log(np.clip(
            self.model.predict(xgb.DMatrix(np.asarray(X, dtype=float))), 1e-6, None))
        z = norm.ppf(np.asarray(qs, dtype=float))
        return np.exp(centre[:, None] + self.scale * z[None, :])


class WeibullAFTIncumbent:
    """The incumbent duration model, refitted here. censoring = modelled."""

    censoring = "modelled"

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, int(seed)
        self.model = None
        self.columns: list[str] = []

    def fit(self, X, duration, observed):
        from lifelines import WeibullAFTFitter

        frame = pd.DataFrame(np.asarray(X, dtype=float),
                             columns=[f"x{i}" for i in range(np.shape(X)[1])])
        # lifelines inverts the design matrix, so perfectly collinear or
        # constant columns are a hard failure rather than a warning. The
        # incumbent drops them in fit_duration; the same has to happen here or
        # this row fails for a linear-algebra reason and gets misread as the
        # model class being unsuitable.
        keep = frame.columns[frame.nunique(dropna=False) > 1]
        frame = frame[keep]
        corr = frame.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        frame = frame.drop(columns=[c for c in upper if upper[c].gt(0.999).any()])
        self.columns = list(frame.columns)
        frame = frame.assign(
            duration_=np.clip(np.asarray(duration, dtype=float), 0.5, None),
            observed_=np.asarray(observed).astype(int))
        self.model = WeibullAFTFitter(penalizer=0.1)
        self.model.fit(frame, duration_col="duration_", event_col="observed_")
        return self

    def predict_quantiles(self, X, qs):
        frame = pd.DataFrame(np.asarray(X, dtype=float),
                             columns=[f"x{i}" for i in range(np.shape(X)[1])])
        frame = frame.reindex(columns=self.columns, fill_value=0.0)
        cols = [np.asarray(self.model.predict_percentile(
            frame, p=1.0 - float(q))).ravel() for q in qs]
        out = np.column_stack(cols)
        finite = out[np.isfinite(out)]
        cap = float(finite.max()) if finite.size else 0.0
        return np.sort(np.nan_to_num(out, nan=0.0, posinf=cap), axis=1)


class CensoringIgnoredGBM:
    """Regress log duration and pretend nothing is censored. censoring = IGNORED.

    This row is in the roster to be beaten. Dropping the censoring indicator is
    the most common shortcut when a survival model is inconvenient, and the
    resulting bias -- systematic under-prediction of exactly the long
    restorations that drive customer-hours -- is invisible in a concordance
    index computed on uncensored rows alone. Putting the shortcut on the
    leaderboard next to the censoring-aware learners is the only way to show
    the size of the effect on this dataset instead of asserting it.
    """

    censoring = "IGNORED"

    def __init__(self, cfg, seed):
        self.cfg, self.seed = cfg, int(seed)
        self.inner = None

    def fit(self, X, duration, observed):
        del observed  # the entire point of this row
        from sklearn.ensemble import HistGradientBoostingRegressor

        def factory():
            return HistGradientBoostingRegressor(
                max_iter=int(self.cfg.get("sk_gbm_rounds", 300)), learning_rate=0.05,
                max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False,
                random_state=self.seed)
        self.inner = ResidualDressed(factory, seed=self.seed)
        self.inner.fit(np.asarray(X, dtype=float),
                       np.log(np.clip(np.asarray(duration, dtype=float), 0.5, None)))
        return self

    def predict_quantiles(self, X, qs):
        return np.exp(self.inner.predict_log_quantiles(X, qs))


def _dur_rsf(cfg, seed):
    from sksurv.ensemble import RandomSurvivalForest
    return SksurvModel(lambda: RandomSurvivalForest(
        n_estimators=int(cfg.get("forest_trees", 400)), min_samples_leaf=15,
        max_features="sqrt", random_state=seed, n_jobs=-1))


def _dur_gbsa(cfg, seed):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    return SksurvModel(lambda: GradientBoostingSurvivalAnalysis(
        n_estimators=int(cfg.get("sk_gbm_rounds", 300)), learning_rate=0.05,
        max_depth=3, min_samples_leaf=20, random_state=seed))


def _dur_coxnet(cfg, seed):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv  # noqa: F401  (import proves the package shape)

    class _Scaled(SksurvModel):
        def fit(self, X, duration, observed):
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler().fit(np.asarray(X, dtype=float))
            return super().fit(self._scaler.transform(np.asarray(X, dtype=float)),
                               duration, observed)

        def predict_quantiles(self, X, qs):
            return super().predict_quantiles(
                self._scaler.transform(np.asarray(X, dtype=float)), qs)

    del seed  # coxnet is deterministic
    return _Scaled(lambda: CoxnetSurvivalAnalysis(l1_ratio=0.5, fit_baseline_model=True))


def _dur_xgb_aft(cfg, seed):
    return XGBAft(cfg, seed)


def _dur_weibull(cfg, seed):
    return WeibullAFTIncumbent(cfg, seed)


def _dur_ignored(cfg, seed):
    return CensoringIgnoredGBM(cfg, seed)


DURATION: list[Learner] = [
    Learner("weibull_aft_incumbent", ("lifelines",), _dur_weibull, "parametric",
            censoring="modelled", is_incumbent=True,
            note="incumbent duration model, refitted inside this harness"),
    Learner("random_survival_forest", ("sksurv",), _dur_rsf, "tree",
            censoring="modelled"),
    Learner("gradient_boosted_survival", ("sksurv",), _dur_gbsa, "boost",
            censoring="modelled"),
    Learner("coxnet", ("sksurv",), _dur_coxnet, "linear", censoring="modelled"),
    Learner("xgboost_aft", ("xgboost",), _dur_xgb_aft, "boost",
            censoring="modelled",
            note="interval-censored labels; homoscedastic log-time spread"),
    Learner("gbm_censoring_ignored", ("sklearn",), _dur_ignored, "boost",
            censoring="IGNORED",
            note="deliberate negative control -- quantifies the cost of "
                 "dropping the censoring indicator"),
]


HEADS = {"occurrence": OCCURRENCE, "magnitude": MAGNITUDE, "duration": DURATION}


def roster(head: str, only: list[str] | None = None) -> list[Learner]:
    learners = HEADS[head]
    if only:
        wanted = set(only)
        learners = [ln for ln in learners if ln.name in wanted]
        unknown = wanted - {ln.name for ln in HEADS[head]}
        if unknown:
            raise SystemExit(f"unknown {head} learner(s): {sorted(unknown)}. "
                             f"Available: {[ln.name for ln in HEADS[head]]}")
    return learners
