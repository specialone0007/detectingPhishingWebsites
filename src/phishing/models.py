"""The model zoo. One factory per model name; hyper-parameters are the ones that came out
of the 2022 tuning runs, re-checked with 5-fold CV in `scripts/run_experiments.py`.

All estimators are scikit-learn compatible so the evaluation code treats them uniformly.
The features are already ordinal in {-1, 0, 1}; the SVMs and the MLP get a StandardScaler
in front, the tree ensembles take the raw values.
"""

from __future__ import annotations

from collections.abc import Callable

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def _scaled(est: BaseEstimator) -> Pipeline:
    return make_pipeline(StandardScaler(), est)


MODELS: dict[str, Callable[[int], BaseEstimator]] = {
    # ---- baselines -------------------------------------------------------------
    "logistic": lambda seed: _scaled(LogisticRegression(C=1.0, max_iter=2000, random_state=seed)),
    # ---- from the 2022 report --------------------------------------------------
    "svm_linear": lambda seed: _scaled(
        SVC(kernel="linear", C=0.1, probability=True, random_state=seed)
    ),
    "svm_rbf": lambda seed: _scaled(
        SVC(kernel="rbf", C=100, gamma=0.125, probability=True, random_state=seed)
    ),
    "mlp": lambda seed: _scaled(
        MLPClassifier(hidden_layer_sizes=(64, 64, 32), alpha=1e-3, max_iter=600,
                      early_stopping=True, random_state=seed)
    ),
    "xgboost": lambda seed: XGBClassifier(
        n_estimators=1000, learning_rate=0.1, gamma=1, max_depth=6,
        reg_lambda=0.15, reg_alpha=0.15, max_bin=1024,
        eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
    ),
    "lightgbm": lambda seed: LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=10, num_leaves=63,
        extra_trees=True, feature_fraction=0.9, bagging_freq=8, bagging_fraction=0.8,
        reg_alpha=0.15, reg_lambda=0.15, random_state=seed, n_jobs=-1, verbose=-1,
    ),
    # ---- LightGBM with library defaults, to check whether the 2022 tuning helped ---
    "lightgbm_default": lambda seed: LGBMClassifier(
        n_estimators=500, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1
    ),
    # ---- one more strong, cheap ensemble for reference ---------------------------
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=seed
    ),
}

PRETTY = {
    "logistic": "Logistic regression",
    "svm_linear": "SVM (linear)",
    "svm_rbf": "SVM (RBF)",
    "mlp": "Neural network (MLP)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM (2022 params)",
    "lightgbm_default": "LightGBM (defaults)",
    "random_forest": "Random forest",
}


def make_model(name: str, seed: int = 42) -> BaseEstimator:
    try:
        return MODELS[name](seed)
    except KeyError as e:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from e
